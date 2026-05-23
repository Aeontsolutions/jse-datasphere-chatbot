"""Tests for PersonaActor — the LLM that role-plays the user persona."""

from unittest.mock import MagicMock

import pytest

from evals.persona import PersonaSpec
from evals.persona_actor import PersonaActor, PersonaTurn


def _make_persona() -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="A skeptical analyst.",
        goal="Find NCB revenue growth.",
        max_turns=4,
    )


def _mock_genai_response(text_json: str) -> MagicMock:
    response = MagicMock()
    response.text = text_json
    return response


def _async_value(value):
    """Coerce a sync value into an awaitable returning that value."""
    async def _coro():
        return value

    return _coro()


@pytest.mark.asyncio
async def test_act_returns_structured_turn():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock()
    fake_client.aio.models.generate_content.return_value = _async_value(
        _mock_genai_response('{"utterance": "Show me NCB revenue.", "done": false, "done_reason": null}')
    )

    actor = PersonaActor(
        client=fake_client,
        model="gemini-2.5-flash",
        temperature=0.8,
    )
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=0)

    assert isinstance(turn, PersonaTurn)
    assert turn.utterance == "Show me NCB revenue."
    assert turn.done is False
    assert turn.done_reason is None


@pytest.mark.asyncio
async def test_act_parses_done_signal():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock()
    fake_client.aio.models.generate_content.return_value = _async_value(
        _mock_genai_response(
            '{"utterance": "Thanks, that\'s clear.", "done": true, "done_reason": "got the breakdown"}'
        )
    )

    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=1)

    assert turn.done is True
    assert turn.done_reason == "got the breakdown"


@pytest.mark.asyncio
async def test_act_retries_once_on_malformed_json():
    persona = _make_persona()
    fake_client = MagicMock()
    call_results = [
        _mock_genai_response("not json at all"),
        _mock_genai_response('{"utterance": "Recovery question.", "done": false, "done_reason": null}'),
    ]
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[_async_value(r) for r in call_results]
    )

    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=0)
    assert turn.utterance == "Recovery question."


@pytest.mark.asyncio
async def test_act_raises_after_second_malformed_json():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("still not json")),
            _async_value(_mock_genai_response("still still not json")),
        ]
    )
    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    with pytest.raises(RuntimeError, match="persona_malformed"):
        await actor.act(persona=persona, transcript_history=[], replicate_index=0)
