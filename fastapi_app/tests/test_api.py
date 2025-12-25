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


if __name__ == "__main__":
    test_api()
