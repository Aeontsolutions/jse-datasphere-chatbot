"""LLM persona actor — role-plays the user side of a simulated conversation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from evals.persona import PersonaSpec


class PersonaTurn(BaseModel):
    """One persona-side utterance plus the optional done signal."""

    utterance: str
    done: bool
    done_reason: str | None = None


_SYSTEM_TEMPLATE = """You are role-playing a user interacting with a financial chatbot.

# Your character
{character}

# Your goal
{goal}

# Rules
- Stay in character. Use your character's voice, vocabulary, and level of detail.
- Each turn, respond with ONE message you would send to the chatbot.
- Set "done": true ONLY when you believe your goal has been satisfied,
  OR you have decided it cannot be (and explain in done_reason).
- Do not pretend to be the chatbot. Do not narrate. Only produce what you
  would actually type as a user.
- Output ONLY JSON of the schema below.

# Output JSON schema
{{"utterance": "<the message you send>", "done": <true|false>, "done_reason": "<short reason or null>"}}
"""


def _seed_for(persona_id: str, replicate_index: int) -> int:
    digest = hashlib.sha256(persona_id.encode("utf-8")).digest()
    base = int.from_bytes(digest[:4], "big")
    return (base + replicate_index) & 0x7FFFFFFF


def _format_history(turns: list[dict[str, str]]) -> str:
    """Format prior turns the persona can see (text only, no metadata)."""
    if not turns:
        return "(this is your first message — no history yet)"
    lines = []
    for t in turns:
        lines.append(f"You: {t['persona_utterance']}")
        lines.append(f"Bot: {t['chatbot_text']}")
    return "\n".join(lines)


class PersonaActor:
    """Wraps a google-genai client to act as the persona."""

    def __init__(self, client: Any, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def act(
        self,
        persona: PersonaSpec,
        transcript_history: list[dict[str, str]],
        replicate_index: int,
    ) -> PersonaTurn:
        system_text = _SYSTEM_TEMPLATE.format(
            character=persona.character.strip(),
            goal=persona.goal.strip(),
        )
        history_text = _format_history(transcript_history)
        seed = _seed_for(persona.id, replicate_index)

        for attempt in range(2):
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    {"role": "user", "parts": [{"text": system_text + "\n\n" + history_text}]}
                ],
                config={
                    "temperature": self._temperature if attempt == 0 else 0.0,
                    "response_mime_type": "application/json",
                    "seed": seed,
                },
            )
            try:
                data = json.loads(response.text)
                return PersonaTurn(**data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        raise RuntimeError(
            f"persona_malformed: persona {persona.id} returned unparseable JSON twice"
        )
