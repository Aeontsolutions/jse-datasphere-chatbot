"""LLM judge — scores a completed conversation against persona expectations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from evals._genai_retry import call_with_retry
from evals.metrics import estimate_gemini_cost_usd, usage_tokens_from_response
from evals.persona import PersonaSpec
from evals.transcript import Transcript

_METADATA_CHAR_LIMIT = 4000

DEFAULT_RUBRIC_PATH = Path(__file__).parent / "config" / "judge_rubric.yaml"


class DimensionScore(BaseModel):
    score: int | None
    justification: str


class FactfulnessScore(BaseModel):
    score: int | None
    facts_satisfied: list[bool] = Field(default_factory=list)
    justification: str


class ToolUseScore(BaseModel):
    score: int
    observed_tools: list[str] = Field(default_factory=list)
    justification: str


class JudgeScores(BaseModel):
    groundedness: DimensionScore
    factfulness: FactfulnessScore
    goal_completion: DimensionScore
    tool_use_appropriateness: ToolUseScore
    coherence: DimensionScore
    persona_handling: DimensionScore


class NotableMoment(BaseModel):
    turn: int
    type: str
    note: str


class JudgeOutput(BaseModel):
    scores: JudgeScores
    verdict: Literal["pass", "fail", "partial"]
    verdict_reason: str
    notable_moments: list[NotableMoment] = Field(default_factory=list)
    cost_usd: float = 0.0
    """Cost of the judge's Gemini call. Not part of the LLM's own JSON output --
    set by Judge.evaluate() after parsing."""
    truncated_turn_count: int = 0
    """Number of turns whose chatbot_metadata was too large to show the judge
    in full. Not part of the LLM's own JSON output -- set by Judge.evaluate()."""


def load_rubric(path: Path) -> dict[str, Any]:
    """Load judge_rubric.yaml. Public so report.py can reuse `verdict_weights`
    for the weighted-verdict cross-check without re-implementing YAML loading."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


_PROMPT_TEMPLATE = """You are an expert evaluator of a financial chatbot.

# Current date
Today's real-world date is {today}. Your own training data has an earlier
cutoff -- do not use it to judge plausibility. A claim citing a report,
filing, or market event dated on or before {today} is NOT "impossible",
"hypothetical", or "future" just because it postdates what you were trained
on. Only mark a claim as ungrounded or hallucinated when it conflicts with
the transcript's own sources/metadata below, never because a date in it
feels futuristic to you.

# Persona under test
- id: {persona_id}
- category: {persona_category}
- character: {character}
- goal: {goal}
- expected_facts:
{expected_facts_block}

# Endpoint hit
{endpoint}

# Run metadata
- total_turns: {total_turns}
- termination: {termination}
- total_latency_ms: {total_latency_ms}
- total_cost_usd: {total_cost_usd}

# Full transcript (each turn includes the chatbot's raw API metadata so you can
# verify groundedness against `sources`, `tools_executed`, `filters_used`, etc.)
{transcript_block}

# Scoring rubric
{rubric_block}

# Output instructions
Score each dimension on a 1–5 integer scale. For `factfulness`, set score to
null and facts_satisfied to [] if expected_facts is empty.
Produce strictly valid JSON matching this schema:

{{
  "scores": {{
    "groundedness": {{"score": int, "justification": str}},
    "factfulness": {{"score": int | null, "facts_satisfied": [bool], "justification": str}},
    "goal_completion": {{"score": int, "justification": str}},
    "tool_use_appropriateness": {{"score": int, "observed_tools": [str], "justification": str}},
    "coherence": {{"score": int, "justification": str}},
    "persona_handling": {{"score": int, "justification": str}}
  }},
  "verdict": "pass" | "fail" | "partial",
  "verdict_reason": str,
  "notable_moments": [{{"turn": int, "type": str, "note": str}}]
}}

Output ONLY the JSON, no commentary.
"""


def _format_transcript(transcript: Transcript) -> tuple[str, list[int]]:
    """Render the transcript for the judge prompt.

    Returns (text, truncated_turn_indices). Turns whose metadata JSON exceeds
    _METADATA_CHAR_LIMIT are cut but visibly marked -- both to the judge (so
    it doesn't score groundedness against data it can't see) and to callers
    (so report.py can surface how often this happens).
    """
    lines = []
    truncated_turns: list[int] = []
    for t in transcript.turns:
        lines.append(f"--- Turn {t.turn_index} ---")
        lines.append(f"USER: {t.persona_utterance}")
        lines.append(f"BOT TEXT: {t.chatbot_text}")
        metadata_json = json.dumps(t.chatbot_metadata, indent=2)
        if len(metadata_json) > _METADATA_CHAR_LIMIT:
            omitted = len(metadata_json) - _METADATA_CHAR_LIMIT
            metadata_json = (
                f"{metadata_json[:_METADATA_CHAR_LIMIT]}\n"
                f"...[TRUNCATED: {omitted} chars omitted -- do not assume claims "
                f"beyond this point are unsupported, but treat groundedness here "
                f"as unverifiable rather than confirmed]"
            )
            truncated_turns.append(t.turn_index)
        lines.append(f"BOT METADATA (sources, tools, filters):\n{metadata_json}")
    return "\n".join(lines), truncated_turns


def _format_facts(facts: list[str]) -> str:
    if not facts:
        return "  (none — set factfulness.score to null)"
    return "\n".join(f"  - {f}" for f in facts)


def _format_rubric(rubric: dict[str, Any]) -> str:
    dims = rubric.get("dimensions") or {}
    parts: list[str] = []
    for name, body in dims.items():
        parts.append(f"## {name}\n{body.get('description', '').strip()}")
    return "\n\n".join(parts)


class Judge:
    """Wraps the google-genai client to score one conversation."""

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        rubric_path: Path | None = None,
        today: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._rubric = load_rubric(rubric_path or DEFAULT_RUBRIC_PATH)
        # Anchors the judge against its own training cutoff. Captured once per
        # Judge instance (one per eval run) rather than per evaluate() call,
        # since day-level granularity makes re-computing it per-conversation
        # pointless.
        self._today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _generate(self, prompt: str, temperature: float) -> Any:
        return await self._client.aio.models.generate_content(
            model=self._model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )

    async def evaluate(
        self,
        persona: PersonaSpec,
        transcript: Transcript,
    ) -> JudgeOutput:
        totals = transcript.totals()
        transcript_block, truncated_turns = _format_transcript(transcript)
        prompt = _PROMPT_TEMPLATE.format(
            today=self._today,
            persona_id=persona.id,
            persona_category=persona.category,
            character=persona.character.strip(),
            goal=persona.goal.strip(),
            expected_facts_block=_format_facts(persona.expected_facts),
            endpoint=persona.endpoint,
            total_turns=totals["turns"],
            termination=transcript.termination.reason,
            total_latency_ms=int(totals["latency_ms"]),
            total_cost_usd=round(totals["cost_usd"], 6),
            transcript_block=transcript_block,
            rubric_block=_format_rubric(self._rubric),
        )

        for attempt in range(2):
            # Each parse attempt retries transient Gemini errors of its own.
            # A 503 means the model is busy, not that the judge failed -- and
            # an unjudged conversation is dropped from the scored sample, so
            # letting one through quietly shrinks the evidence a release rests
            # on. See evals/_genai_retry.py.
            response = await call_with_retry(
                partial(
                    self._generate,
                    prompt,
                    self._temperature if attempt == 0 else 0.0,
                ),
                label=f"judge:{transcript.conversation_id}",
            )
            try:
                data = json.loads(response.text)
                data.pop("cost_usd", None)  # only ever set by us, never trust the LLM's JSON
                data.pop("truncated_turn_count", None)
                output = JudgeOutput(**data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            input_tokens, output_tokens = usage_tokens_from_response(response)
            output.cost_usd = estimate_gemini_cost_usd(self._model, input_tokens, output_tokens)
            output.truncated_turn_count = len(truncated_turns)
            return output

        raise RuntimeError(
            f"judge_failed: conversation {transcript.conversation_id} judge returned unparseable JSON twice"
        )
