#!/usr/bin/env python3
"""
Simple test script for the streaming chat endpoints.
This script tests both the traditional chat streaming and fast chat streaming endpoints.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"


def test_streaming_endpoint(endpoint_path, test_query="What are the main financial highlights?"):
    """Test a streaming endpoint and print progress updates."""

    print(f"\n{'='*60}")
    print(f"Testing: {endpoint_path}")
    print(f"Query: {test_query}")
    print(f"{'='*60}")

    url = f"{BASE_URL}{endpoint_path}"

    request_data = {
        "query": test_query,
        "auto_load_documents": True,
        "memory_enabled": True,
        "conversation_history": [],
    }

    start_time = time.time()

    try:
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        print("✅ Connection established, receiving stream...")

        current_event = ""
        buffer = ""

        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                buffer += chunk
                lines = buffer.split("\n")

                # Process complete lines, keep incomplete line in buffer
                buffer = lines[-1] if not lines[-1].endswith("\n") else ""

                for line in lines[:-1] if not lines[-1].endswith("\n") else lines:
                    line = line.strip()

                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        data_str = line[6:]
                        if data_str and data_str != "{}":  # Ignore empty heartbeats
                            try:
                                data = json.loads(data_str)
                                handle_stream_event(current_event, data)

                                # Break on final result or error
                                if current_event in ["result", "error"]:
                                    print(
                                        f"\n⏱️  Total time: {time.time() - start_time:.2f} seconds"
                                    )
                                    return data

                            except json.JSONDecodeError as e:
                                print(f"⚠️  JSON decode error: {e}")
                                print(f"    Data: {data_str}")

        print(f"\n⏱️  Total time: {time.time() - start_time:.2f} seconds")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def handle_stream_event(event, data):
    """Handle different types of stream events."""

    if event == "progress":
        step = data.get("step", "unknown")
        message = data.get("message", "")
        progress = data.get("progress", 0)
        details = data.get("details", {})

        # Format progress bar
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        print(f"📊 [{bar}] {progress:5.1f}% | {step}: {message}")

        if details:
            details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            print(f"    📋 Details: {details_str}")

    elif event == "result":
        print("\n🎉 SUCCESS! Response received:")
        print(f"    💬 Response length: {len(data.get('response', ''))} characters")

        # Handle different response types
        if "documents_loaded" in data:
            # Document-based response
            docs_loaded = data.get("documents_loaded", [])
            if docs_loaded:
                print(f"    📚 Documents loaded: {len(docs_loaded)}")
                for doc in docs_loaded[:3]:  # Show first 3
                    print(f"       - {doc}")
                if len(docs_loaded) > 3:
                    print(f"       ... and {len(docs_loaded) - 3} more")

            selection_msg = data.get("document_selection_message")
            if selection_msg:
                print(f"    🎯 Selection: {selection_msg}")
        elif "data_found" in data:
            # Financial data response
            print(f"    📊 Data found: {data.get('data_found', False)}")
            print(f"    📈 Records returned: {data.get('record_count', 0)}")

            filters = data.get("filters_used", {})
            if filters:
                companies = filters.get("companies", [])
                years = filters.get("years", [])
                if companies:
                    print(f"    🏢 Companies: {', '.join(companies)}")
                if years:
                    print(f"    📅 Years: {', '.join(years)}")

            warnings = data.get("warnings", [])
            if warnings:
                print(f"    ⚠️  Warnings: {len(warnings)}")
                for warning in warnings[:2]:  # Show first 2 warnings
                    print(f"       - {warning}")

        # Show first part of response
        response_text = data.get("response", "")
        if response_text:
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"    📝 Response preview: {preview}")

    elif event == "error":
        error_msg = data.get("error", "Unknown error")
        print(f"\n❌ ERROR: {error_msg}")

    elif event == "heartbeat":
        print("💓", end="", flush=True)
    else:
        print(f"❓ Unknown event '{event}': {data}")


def main():
    """Main test function."""

    print("🚀 Starting streaming chat endpoint tests...")

    # Test query
    test_query = "What are the main financial highlights for MTN Group?"

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print(f"📝 Using query: '{test_query}'")

    # Test all streaming endpoints
    endpoints = [
        "/chat/stream",
        "/fast_chat/stream",
        "/fast_chat_v2/stream",  # New financial streaming endpoint
    ]

    for endpoint in endpoints:
        result = test_streaming_endpoint(endpoint, test_query)

        if result is None:
            print(f"❌ Test failed for {endpoint}")
        else:
            print(f"✅ Test completed for {endpoint}")

        # Small delay between tests
        time.sleep(1)

    print(f"\n{'='*60}")
    print("🏁 All tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
