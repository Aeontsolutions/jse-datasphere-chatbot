"""Model-selection and malformed-request guards for the Gemini 3.7 migration.

Covers three things the migration depends on:

1. Every model gets the lowest-thinking config it actually honours. The knob is
   a per-model contract -- measured 2026-08-29, `thinking_budget=0` is a 400 on
   3.6 and is silently ignored by 3.7, while `thinking_level="MINIMAL"` is a 400
   on 3.7. Getting this wrong is either an outage or invisible token spend.
2. The model for each phase stays overridable by environment variable, so a
   candidate can be trialled in one environment without a code change.
3. Contents never end on a model turn. 2.5 tolerated that; every 3.x model
   rejects it with 400 INVALID_ARGUMENT (issue #82).
"""

import pytest

from app.agent_v2 import (
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SYNTHESIS_MODEL,
    AgentV2,
    resolve_no_thinking,
    resolve_router_model,
    resolve_synthesis_model,
)


class TestDefaults:
    def test_router_and_synthesis_default_to_3_7(self):
        assert DEFAULT_ROUTER_MODEL == "gemini-3.7-flash"
        assert DEFAULT_SYNTHESIS_MODEL == "gemini-3.7-flash"


class TestThinkingFloor:
    """The measured per-model contract. See the table in agent_v2.py."""

    def test_3_7_uses_thinking_level_low(self):
        # MINIMAL is rejected by 3.7 with 400 "not supported for this model".
        config = resolve_no_thinking("gemini-3.7-flash")
        assert config.thinking_level == "LOW"
        assert config.thinking_budget is None

    def test_3_6_uses_thinking_level_minimal(self):
        # 3.6 rejects thinking_budget=0 outright, and MINIMAL costs it 0 tokens.
        config = resolve_no_thinking("gemini-3.6-flash")
        assert config.thinking_level == "MINIMAL"
        assert config.thinking_budget is None

    def test_2_5_uses_thinking_budget_zero(self):
        config = resolve_no_thinking("gemini-2.5-flash")
        assert config.thinking_budget == 0
        assert config.thinking_level is None

    def test_unknown_model_falls_back_to_thinking_budget(self):
        # A model we have not measured must not silently inherit another
        # family's setting -- the pre-3.x default is the documented fallback.
        config = resolve_no_thinking("gemini-9.9-flash")
        assert config.thinking_budget == 0

    def test_match_is_case_insensitive(self):
        assert resolve_no_thinking("GEMINI-3.7-FLASH").thinking_level == "LOW"


class TestEnvOverrides:
    def test_router_model_override(self, monkeypatch):
        monkeypatch.setenv("ROUTER_MODEL_NAME", "gemini-2.5-flash")
        assert resolve_router_model() == "gemini-2.5-flash"

    def test_synthesis_model_override(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIS_MODEL_NAME", "gemini-2.5-pro")
        assert resolve_synthesis_model() == "gemini-2.5-pro"

    def test_blank_override_is_ignored(self, monkeypatch):
        # An empty env var in a manifest must not blank the model name.
        monkeypatch.setenv("SYNTHESIS_MODEL_NAME", "   ")
        assert resolve_synthesis_model() == DEFAULT_SYNTHESIS_MODEL


class TestBuildContents:
    """Issue #82: a contents array must never end on a model turn."""

    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.setattr("app.agent_v2.get_genai_client", lambda: object())
        return AgentV2()

    def test_normal_query_ends_on_user_turn(self, agent):
        contents = agent._build_contents(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "what is NCB revenue?",
        )
        assert contents[-1].role == "user"
        assert contents[-1].parts[0].text == "what is NCB revenue?"

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_query_is_not_appended(self, agent, blank):
        contents = agent._build_contents([{"role": "user", "content": "hi"}], blank)
        assert [c.parts[0].text for c in contents] == ["hi"]

    def test_trailing_model_turns_are_trimmed(self, agent):
        # The exact shape that 400s on 3.x: history ending on an assistant
        # message, with no new user message to follow it.
        contents = agent._build_contents(
            [
                {"role": "assistant", "content": "market summary"},
                {"role": "user", "content": "and 2024?"},
                {"role": "assistant", "content": "2024 summary"},
            ],
            "",
        )
        assert contents[-1].role == "user"
        assert [c.parts[0].text for c in contents] == ["market summary", "and 2024?"]

    def test_history_of_only_model_turns_yields_empty_contents(self, agent):
        contents = agent._build_contents([{"role": "assistant", "content": "hi"}], "")
        assert contents == []


class TestBlankQueryShortCircuit:
    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.setattr("app.agent_v2.get_genai_client", lambda: object())
        return AgentV2()

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.asyncio
    async def test_blank_query_asks_for_a_question_without_calling_the_model(
        self, agent, blank, monkeypatch
    ):
        def explode(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the model must not be called for a blank query")

        monkeypatch.setattr(agent, "_fast_path", explode)

        result = await agent.run(blank)

        assert result["needs_clarification"] is True
        assert result["clarification_question"] == result["response"]
        assert result["data_found"] is False
        assert result["tools_executed"] == []

    @pytest.mark.asyncio
    async def test_blank_query_leaves_history_untouched(self, agent):
        history = [{"role": "user", "content": "hi"}]
        result = await agent.run("", conversation_history=history)
        assert result["conversation_history"] == history
