#!/usr/bin/env python3
"""
Example Load Testing Script for JSE Datasphere Chatbot API

This script demonstrates how to use the load testing tools for quick testing.
Run this to get started with load testing your API endpoints.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any

async def simple_chat_test(base_url: str = "http://localhost:8000"):
    """Simple test for the /chat endpoint"""
    print("🧪 Testing /chat endpoint...")
    
    payload = {
        "query": "What are the key financial metrics for JSE companies?",
        "conversation_history": [],
        "auto_load_documents": True,
        "memory_enabled": True
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        try:
            async with session.post(
                f"{base_url}/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Success! Response time: {response_time:.2f}s")
                    print(f"   Response length: {len(data.get('response', ''))} characters")
                    print(f"   Documents loaded: {len(data.get('documents_loaded', []))}")
                else:
                    print(f"❌ Failed with status: {response.status}")
                    print(f"   Response: {await response.text()}")
                    
        except Exception as e:
            response_time = time.time() - start_time
            print(f"❌ Error: {e}")
            print(f"   Time elapsed: {response_time:.2f}s")

async def simple_financial_test(base_url: str = "http://localhost:8000"):
    """Simple test for the /fast_chat_v2 endpoint"""
    print("🧪 Testing /fast_chat_v2 endpoint...")
    
    payload = {
        "query": "Show me MDS revenue for 2024",
        "conversation_history": [],
        "memory_enabled": True
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        try:
            async with session.post(
                f"{base_url}/fast_chat_v2",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Success! Response time: {response_time:.2f}s")
                    print(f"   Response: {data.get('response', '')[:100]}...")
                    print(f"   Data found: {data.get('data_found', False)}")
                    print(f"   Record count: {data.get('record_count', 0)}")
                else:
                    print(f"❌ Failed with status: {response.status}")
                    print(f"   Response: {await response.text()}")
                    
        except Exception as e:
            response_time = time.time() - start_time
            print(f"❌ Error: {e}")
            print(f"   Time elapsed: {response_time:.2f}s")

async def quick_load_test(base_url: str = "http://localhost:8000", num_requests: int = 10):
    """Quick load test with multiple concurrent requests"""
    print(f"🚀 Running quick load test with {num_requests} concurrent requests...")
    
    payload = {
        "query": "What are the key financial metrics for JSE companies?",
        "conversation_history": [],
        "auto_load_documents": True,
        "memory_enabled": True
    }
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{base_url}/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_time = time.time() - start_time
                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "success": response.status == 200
                    }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "status": 0,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                }
    
    # Create concurrent requests
    tasks = [make_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    successful = sum(1 for r in results if r["success"])
    failed = num_requests - successful
    response_times = [r["response_time"] for r in results if r["success"]]
    
    print(f"\n📊 Quick Load Test Results:")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {(successful/num_requests)*100:.1f}%")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Min response time: {min_time:.2f}s")
        print(f"   Max response time: {max_time:.2f}s")

async def main():
    """Main function to run example tests"""
    print("🚀 JSE Datasphere Chatbot API - Example Load Tests")
    print("=" * 60)
    
    # Check if API is accessible
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/docs") as response:
                if response.status == 200:
                    print(f"✅ API is accessible at {base_url}")
                else:
                    print(f"⚠️  API responded with status {response.status}")
                    print("   Make sure your FastAPI server is running")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to API at {base_url}")
        print(f"   Error: {e}")
        print("   Make sure your FastAPI server is running with:")
        print("   cd fastapi_app && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print("\n" + "=" * 60)
    
    # Run simple tests
    await simple_chat_test(base_url)
    print()
    await simple_financial_test(base_url)
    
    print("\n" + "=" * 60)
    
    # Run quick load test
    await quick_load_test(base_url, num_requests=5)
    
    print("\n" + "=" * 60)
    print("🎉 Example tests completed!")
    print("\nNext steps:")
    print("1. Run comprehensive load tests: python load_test.py")
    print("2. Use Locust for advanced testing: locust -f locustfile.py")
    print("3. Run stress tests: ./stress_test.sh --endpoint chat")
    print("4. Check the LOAD_TESTING_README.md for detailed instructions")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()

