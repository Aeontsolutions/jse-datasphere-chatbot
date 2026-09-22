#!/usr/bin/env python3
"""
Load Testing Script for JSE Datasphere Chatbot API

This script provides comprehensive load testing for:
- /chat endpoint
- /chat/stream endpoint  
- /fast_chat_v2 endpoint
- /fast_chat_v2/stream endpoint

Usage:
    python load_test.py                    # Run basic load test
    python load_test.py --endpoint chat   # Test specific endpoint
    python load_test.py --users 100       # Set number of concurrent users
    python load_test.py --duration 300    # Set test duration in seconds
"""

import asyncio
import aiohttp
import time
import json
import random
import argparse
from typing import List, Dict, Any
import statistics
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: float
    success: bool
    error_message: str = ""

class LoadTester:
    """Load testing class for FastAPI endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results: List[TestResult] = []
        
        # Sample queries for different endpoints
        self.chat_queries = [
            "What are the key financial metrics for JSE companies?",
            "Show me the revenue trends for major companies",
            "What about profit margins in the technology sector?",
            "Compare financial performance across different industries",
            "What are the latest quarterly results?",
            "How do companies perform in different market conditions?",
            "What about dividend yields and payout ratios?",
            "Show me the debt-to-equity ratios",
            "What are the working capital trends?",
            "How do companies manage their cash flow?"
        ]
        
        self.financial_queries = [
            "Show me MDS revenue for 2024",
            "Compare JBG and CPJ profit margins",
            "What about 2022?",
            "Show me the top 5 companies by revenue",
            "What are the average profit margins by industry?",
            "Show me companies with debt ratios above 50%",
            "What about companies with high dividend yields?",
            "Show me the fastest growing companies",
            "Compare 2023 vs 2022 performance",
            "What are the most profitable sectors?"
        ]
        
        # Conversation history templates
        self.conversation_templates = [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you with financial data today?"}],
            [{"role": "user", "content": "Show me some data"}, {"role": "assistant", "content": "I'd be happy to help! What specific financial information are you looking for?"}],
            [],
            [{"role": "user", "content": "What about previous years?"}, {"role": "assistant", "content": "I can show you historical data. Which years would you like to compare?"}]
        ]

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={'Content-Type': 'application/json'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _generate_chat_request(self) -> Dict[str, Any]:
        """Generate a random chat request"""
        return {
            "query": random.choice(self.chat_queries),
            "conversation_history": random.choice(self.conversation_templates),
            "auto_load_documents": random.choice([True, False]),
            "memory_enabled": random.choice([True, False])
        }

    def _generate_financial_request(self) -> Dict[str, Any]:
        """Generate a random financial data request"""
        return {
            "query": random.choice(self.financial_queries),
            "conversation_history": random.choice(self.conversation_templates),
            "memory_enabled": random.choice([True, False])
        }

    async def test_chat_endpoint(self) -> TestResult:
        """Test the /chat endpoint"""
        start_time = time.time()
        try:
            request_data = self._generate_chat_request()
            async with self.session.post(
                f"{self.base_url}/chat",
                json=request_data
            ) as response:
                response_time = time.time() - start_time
                content = await response.text()
                
                return TestResult(
                    endpoint="/chat",
                    method="POST",
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=start_time,
                    success=response.status == 200,
                    error_message="" if response.status == 200 else content
                )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint="/chat",
                method="POST",
                status_code=0,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

    async def test_chat_stream_endpoint(self) -> TestResult:
        """Test the /chat/stream endpoint"""
        start_time = time.time()
        try:
            request_data = self._generate_chat_request()
            async with self.session.post(
                f"{self.base_url}/chat/stream",
                json=request_data
            ) as response:
                response_time = time.time() - start_time
                
                # Read the stream to completion
                content = ""
                async for line in response.content:
                    if line:
                        content += line.decode('utf-8')
                
                return TestResult(
                    endpoint="/chat/stream",
                    method="POST",
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=start_time,
                    success=response.status == 200,
                    error_message="" if response.status == 200 else content
                )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint="/chat/stream",
                method="POST",
                status_code=0,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

    async def test_fast_chat_v2_endpoint(self) -> TestResult:
        """Test the /fast_chat_v2 endpoint"""
        start_time = time.time()
        try:
            request_data = self._generate_financial_request()
            async with self.session.post(
                f"{self.base_url}/fast_chat_v2",
                json=request_data
            ) as response:
                response_time = time.time() - start_time
                content = await response.text()
                
                return TestResult(
                    endpoint="/fast_chat_v2",
                    method="POST",
                    response_time=response_time,
                    timestamp=start_time,
                    success=response.status == 200,
                    error_message="" if response.status == 200 else content
                )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint="/fast_chat_v2",
                method="POST",
                status_code=0,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

    async def test_fast_chat_v2_stream_endpoint(self) -> TestResult:
        """Test the /fast_chat_v2/stream endpoint"""
        start_time = time.time()
        try:
            request_data = self._generate_financial_request()
            async with self.session.post(
                f"{self.base_url}/fast_chat_v2/stream",
                json=request_data
            ) as response:
                response_time = time.time() - start_time
                
                # Read the stream to completion
                content = ""
                async for line in response.content:
                    if line:
                        content += line.decode('utf-8')
                
                return TestResult(
                    endpoint="/fast_chat_v2/stream",
                    method="POST",
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=start_time,
                    success=response.status == 200,
                    error_message="" if response.status == 200 else content
                )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint="/fast_chat_v2/stream",
                method="POST",
                status_code=0,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

    async def run_single_test(self, endpoint: str) -> TestResult:
        """Run a single test for the specified endpoint"""
        if endpoint == "chat":
            return await self.test_chat_endpoint()
        elif endpoint == "chat/stream":
            return await self.test_chat_stream_endpoint()
        elif endpoint == "fast_chat_v2":
            return await self.test_fast_chat_v2_endpoint()
        elif endpoint == "fast_chat_v2/stream":
            return await self.test_fast_chat_v2_stream_endpoint()
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    async def run_load_test(self, endpoint: str, num_users: int, duration: int) -> None:
        """Run a load test with multiple concurrent users"""
        logger.info(f"Starting load test for endpoint: {endpoint}")
        logger.info(f"Concurrent users: {num_users}")
        logger.info(f"Duration: {duration} seconds")
        
        start_time = time.time()
        tasks = []
        
        # Create tasks for concurrent users
        while time.time() - start_time < duration:
            # Create new batch of concurrent requests
            batch_tasks = [
                self.run_single_test(endpoint) 
                for _ in range(min(num_users, 10))  # Limit batch size to avoid overwhelming
            ]
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, TestResult):
                    self.results.append(result)
                else:
                    # Handle exceptions
                    self.results.append(TestResult(
                        endpoint=endpoint,
                        method="POST",
                        status_code=0,
                        response_time=0,
                        timestamp=time.time(),
                        success=False,
                        error_message=str(result)
                    ))
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        logger.info(f"Load test completed. Total requests: {len(self.results)}")

    def print_results(self) -> None:
        """Print test results summary"""
        if not self.results:
            logger.warning("No test results to display")
            return
        
        print("\n" + "="*80)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*80)
        
        # Group results by endpoint
        endpoint_results = {}
        for result in self.results:
            if result.endpoint not in endpoint_results:
                endpoint_results[result.endpoint] = []
            endpoint_results[result.endpoint].append(result)
        
        for endpoint, results in endpoint_results.items():
            print(f"\n{endpoint.upper()} ENDPOINT:")
            print("-" * 50)
            
            total_requests = len(results)
            successful_requests = len([r for r in results if r.success])
            failed_requests = total_requests - successful_requests
            
            response_times = [r.response_time for r in results if r.success]
            
            print(f"Total Requests: {total_requests}")
            print(f"Successful: {successful_requests}")
            print(f"Failed: {failed_requests}")
            print(f"Success Rate: {(successful_requests/total_requests)*100:.2f}%")
            
            if response_times:
                print(f"Average Response Time: {statistics.mean(response_times):.3f}s")
                print(f"Median Response Time: {statistics.median(response_times):.3f}s")
                print(f"Min Response Time: {min(response_times):.3f}s")
                print(f"Max Response Time: {max(response_times):.3f}s")
                print(f"95th Percentile: {statistics.quantiles(response_times, n=20)[18]:.3f}s")
            
            # Show error details if any
            if failed_requests > 0:
                print(f"\nError Summary:")
                error_counts = {}
                for result in results:
                    if not result.success and result.error_message:
                        error_msg = result.error_message[:100] + "..." if len(result.error_message) > 100 else result.error_message
                        error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                
                for error_msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {count}x: {error_msg}")

async def main():
    """Main function to run load tests"""
    parser = argparse.ArgumentParser(description="Load test FastAPI endpoints")
    parser.add_argument("--endpoint", choices=["chat", "chat/stream", "fast_chat_v2", "fast_chat_v2/stream", "all"], 
                       default="all", help="Endpoint to test")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.users <= 0:
        print("Error: Number of users must be positive")
        return
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        return
    
    print(f"Starting load test for: {args.endpoint}")
    print(f"Base URL: {args.url}")
    print(f"Concurrent users: {args.users}")
    print(f"Duration: {args.duration} seconds")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async with LoadTester(args.url) as tester:
        if args.endpoint == "all":
            # Test all endpoints
            endpoints = ["chat", "chat/stream", "fast_chat_v2", "fast_chat_v2/stream"]
            for endpoint in endpoints:
                print(f"\nTesting {endpoint}...")
                await tester.run_load_test(endpoint, args.users, args.duration // len(endpoints))
        else:
            # Test specific endpoint
            await tester.run_load_test(args.endpoint, args.users, args.duration)
        
        # Print results
        tester.print_results()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
    except Exception as e:
        print(f"Error running load test: {e}")
        import traceback
        traceback.print_exc()
