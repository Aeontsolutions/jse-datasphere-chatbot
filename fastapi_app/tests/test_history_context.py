import pytest
from app.financial_utils import FinancialDataManager


@pytest.fixture
def mock_llm():
    class MockLLM:
        def generate_content(self, prompt):
            class Response:
                text = '{"companies": ["Elite Diagnostic Limited"], "symbols": ["ELITE"], "years": ["2024"], "standard_items": ["revenue"], "interpretation": "Elite Diagnostic Limited revenue for 2024", "data_availability_note": "", "is_follow_up": true, "context_used": "conversation history"}'

            return Response()

    return MockLLM()


def test_parse_user_query_with_history(monkeypatch):
    """Test that conversation history is passed to the LLM."""

    # Set required environment variables
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("BIGQUERY_DATASET", "test-dataset")
    monkeypatch.setenv("BIGQUERY_TABLE", "test-table")

    # Mock the LLM to capture the prompt and ensure history is included
    captured_prompt = []

    class MockLLM:
        def generate_content(self, prompt):
            captured_prompt.append(prompt)

            class Response:
                text = '{"companies": ["Elite Diagnostic Limited"], "symbols": ["ELITE"], "years": ["2024"], "standard_items": ["revenue"], "interpretation": "Elite Diagnostic Limited revenue for 2024", "is_follow_up": true}'

            return Response()

    monkeypatch.setattr(
        FinancialDataManager, "_initialize_ai_model", lambda self: setattr(self, "model", MockLLM())
    )

    manager = FinancialDataManager()
    manager.model = MockLLM()
    manager.metadata = {"companies": [], "symbols": [], "years": [], "standard_items": []}

    history = [
        {"role": "user", "content": "Show me Elite revenue 2023"},
        {"role": "assistant", "content": "Here is the data..."},
    ]
    query = "What about 2024?"

    manager.parse_user_query(query, conversation_history=history)

    assert len(captured_prompt) > 0
    # Verify history is in the prompt
    assert "Show me Elite revenue 2023" in captured_prompt[0]
    assert "Here is the data..." in captured_prompt[0]
