from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def financial_manager():
    fm = MagicMock()
    fm.metadata = {
        "symbols": ["NCB", "GK"],
        "years": ["2022", "2023"],
        "associations": {"symbol_to_company": {"NCB": ["NCB Financial Group"]}},
    }
    return fm


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_appends_new_turn_to_conversation_history(financial_manager):
    """AgentOrchestrator.run() must append the new user+assistant turn to history."""
    from app.agent import AgentOrchestrator

    orchestrator = AgentOrchestrator(financial_manager=financial_manager)
    prior_history = [
        {"role": "user", "content": "What is NCB's revenue?"},
        {"role": "assistant", "content": "NCB revenue was $50B."},
    ]

    with (
        patch.object(
            orchestrator,
            "_smart_optimize",
            new_callable=AsyncMock,
            return_value={
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": "NCB profit 2023",
                "routing": {"use_financial": False, "use_web_search": True},
                "defaults_applied": [],
                "resolved_years": [],
            },
        ),
        patch.object(
            orchestrator,
            "_execute_web_search",
            new_callable=AsyncMock,
            return_value={
                "search_results": {},
                "sources": [],
                "context": "NCB profit context from web",
            },
        ),
        patch.object(
            orchestrator,
            "_synthesize",
            new_callable=AsyncMock,
            return_value="NCB's 2023 profit was $10B.",
        ),
        patch.object(orchestrator, "_track_cost"),
    ):
        result = await orchestrator.run(
            query="What about profit?",
            conversation_history=prior_history,
            enable_web_search=True,
            enable_financial_data=False,
        )

    history = result["conversation_history"]
    assert history[-2] == {"role": "user", "content": "What about profit?"}
    assert history[-1] == {"role": "assistant", "content": "NCB's 2023 profit was $10B."}
    assert len(history) == 4  # 2 prior + 2 new


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_empty_history_creates_new_history(financial_manager):
    """AgentOrchestrator.run() with no prior history returns a two-entry history."""
    from app.agent import AgentOrchestrator

    orchestrator = AgentOrchestrator(financial_manager=financial_manager)

    with (
        patch.object(
            orchestrator,
            "_smart_optimize",
            new_callable=AsyncMock,
            return_value={
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": "NCB revenue",
                "routing": {"use_financial": False, "use_web_search": True},
                "defaults_applied": [],
                "resolved_years": [],
            },
        ),
        patch.object(
            orchestrator,
            "_execute_web_search",
            new_callable=AsyncMock,
            return_value={"search_results": {}, "sources": [], "context": "some context"},
        ),
        patch.object(
            orchestrator,
            "_synthesize",
            new_callable=AsyncMock,
            return_value="NCB revenue is $50B.",
        ),
        patch.object(orchestrator, "_track_cost"),
    ):
        result = await orchestrator.run(
            query="NCB revenue",
            conversation_history=None,
            enable_web_search=True,
            enable_financial_data=False,
        )

    history = result["conversation_history"]
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "NCB revenue"}
    assert history[1] == {"role": "assistant", "content": "NCB revenue is $50B."}
