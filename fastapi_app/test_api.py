import requests
import json

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
            "memory_enabled": True
        }
        response = requests.post("http://localhost:8000/chat", json=data)
        print(f"Chat endpoint response: {response.status_code}")
        if response.status_code == 200:
            print("Chat endpoint test successful")
        else:
            print(f"Chat endpoint error: {response.text}")
    except Exception as e:
        print(f"Error testing chat endpoint: {str(e)}")

if __name__ == "__main__":
    test_api()
