import pytest
import requests


# Test the API locally
def test_api():
    print("Testing API endpoints...")

    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health endpoint response: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error testing health endpoint: {str(e)}")

    # Test chat endpoint with a simple query
    try:
        data = {
            "query": "What is the revenue for Company X in 2023?",
            "auto_load_documents": True,
            "memory_enabled": True,
        }
        response = requests.post("http://localhost:8000/chat", json=data)
        print(f"Chat endpoint response: {response.status_code}")
        if response.status_code == 200:
            print("Chat endpoint test successful")
        else:
            print(f"Chat endpoint error: {response.text}")
    except Exception as e:
        print(f"Error testing chat endpoint: {str(e)}")

    # Test cache status endpoint
    try:
        response = requests.get("http://localhost:8000/cache/status")
        print(f"Cache status endpoint response: {response.status_code}")
        if response.status_code == 200:
            print("Cache status response:", response.json())
        else:
            print(f"Cache status error: {response.text}")
    except Exception as e:
        print(f"Error testing cache status endpoint: {str(e)}")

    # Test cache refresh endpoint
    try:
        response = requests.post("http://localhost:8000/cache/refresh")
        print(f"Cache refresh endpoint response: {response.status_code}")
        if response.status_code == 200:
            print("Cache refresh successful")
            print("Cache refresh response:", response.json())
        else:
            print(f"Cache refresh error: {response.text}")
    except Exception as e:
        print(f"Error testing cache refresh endpoint: {str(e)}")


@pytest.mark.unit
def test_chat_stream_passes_enable_financial_data_to_orchestrator(test_client, monkeypatch):
    """
    /chat/stream must instantiate a fresh AgentOrchestrator per request and pass
    enable_financial_data from the request body.
    """
    from unittest.mock import AsyncMock, MagicMock
    import app.main as main_module
    from app.main import app

    mock_result = {
        "response": "NCB 2023 revenue was $50B.",
        "data_found": True,
        "record_count": 3,
        "filters_used": None,
        "data_preview": None,
        "conversation_history": [
            {"role": "user", "content": "NCB revenue 2023"},
            {"role": "assistant", "content": "NCB 2023 revenue was $50B."},
        ],
        "warnings": None,
        "suggestions": None,
        "chart": None,
        "sources": None,
        "web_search_results": None,
        "tools_executed": ["query_financial_data"],
        "needs_clarification": False,
        "clarification_question": None,
        "cost_summary": None,
    }

    mock_orchestrator = MagicMock()
    mock_orchestrator.run = AsyncMock(return_value=mock_result)

    monkeypatch.setattr(app.state, "financial_manager", MagicMock())
    monkeypatch.setattr(main_module, "AgentOrchestrator", lambda financial_manager: mock_orchestrator)

    response = test_client.post(
        "/chat/stream",
        json={
            "query": "NCB revenue 2023",
            "enable_financial_data": True,
            "enable_web_search": False,
            "memory_enabled": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 3
    assert data["tools_executed"] == ["query_financial_data"]

    mock_orchestrator.run.assert_called_once_with(
        query="NCB revenue 2023",
        conversation_history=None,
        enable_web_search=False,
        enable_financial_data=True,
    )


if __name__ == "__main__":
    test_api()
