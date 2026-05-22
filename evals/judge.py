"""LLM judge — scores a completed conversation against persona expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from evals.persona import PersonaSpec
from evals.transcript import Transcript

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


def _load_rubric(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


_PROMPT_TEMPLATE = """You are an expert evaluator of a financial chatbot.

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


def _format_transcript(transcript: Transcript) -> str:
    lines = []
    for t in transcript.turns:
        lines.append(f"--- Turn {t.turn_index} ---")
        lines.append(f"USER: {t.persona_utterance}")
        lines.append(f"BOT TEXT: {t.chatbot_text}")
        lines.append(
            f"BOT METADATA (sources, tools, filters):\n{json.dumps(t.chatbot_metadata, indent=2)[:4000]}"
        )
    return "\n".join(lines)


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
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._rubric = _load_rubric(rubric_path or DEFAULT_RUBRIC_PATH)

    async def evaluate(
        self,
        persona: PersonaSpec,
        transcript: Transcript,
    ) -> JudgeOutput:
        totals = transcript.totals()
        prompt = _PROMPT_TEMPLATE.format(
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
            transcript_block=_format_transcript(transcript),
            rubric_block=_format_rubric(self._rubric),
        )

        for attempt in range(2):
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "temperature": self._temperature if attempt == 0 else 0.0,
                    "response_mime_type": "application/json",
                },
            )
            try:
                data = json.loads(response.text)
                return JudgeOutput(**data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        raise RuntimeError(
            f"judge_failed: conversation {transcript.conversation_id} judge returned unparseable JSON twice"
        )
