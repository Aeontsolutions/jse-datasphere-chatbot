#!/usr/bin/env python3
"""
Locust Load Testing File for JSE Datasphere Chatbot API

This file provides Locust-based load testing for the FastAPI endpoints.
Locust is a popular open-source load testing tool that provides a web UI.

Installation:
    pip install locust

Usage:
    locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 in your browser to access the Locust web UI.
"""

import json
import random
from locust import HttpUser, task, between, events
from typing import Dict, Any, List

class ChatUser(HttpUser):
    """Simulates a user interacting with chat endpoints"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.conversation_history = []
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
    
    @task(3)  # Weight: 3 - more frequent
    def test_chat_endpoint(self):
        """Test the /chat endpoint"""
        query = random.choice(self.chat_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "auto_load_documents": random.choice([True, False]),
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="POST /chat"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Update conversation history if memory is enabled
                    if payload.get("memory_enabled") and data.get("conversation_history"):
                        self.conversation_history = data["conversation_history"]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)  # Weight: 2 - medium frequency
    def test_chat_stream_endpoint(self):
        """Test the /chat/stream endpoint"""
        query = random.choice(self.chat_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "auto_load_documents": random.choice([True, False]),
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/chat/stream",
            json=payload,
            catch_response=True,
            name="POST /chat/stream"
        ) as response:
            if response.status_code == 200:
                # Read the stream content
                content = response.content.decode('utf-8')
                if content:
                    response.success()
                else:
                    response.failure("Empty stream response")
            else:
                response.failure(f"Status code: {response.status_code}")

class FinancialUser(HttpUser):
    """Simulates a user interacting with financial data endpoints"""
    
    wait_time = between(2, 5)  # Wait 2-5 seconds between requests (financial queries are slower)
    
    def on_start(self):
        """Initialize user session"""
        self.conversation_history = []
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
    
    @task(3)  # Weight: 3 - more frequent
    def test_fast_chat_v2_endpoint(self):
        """Test the /fast_chat_v2 endpoint"""
        query = random.choice(self.financial_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/fast_chat_v2",
            json=payload,
            catch_response=True,
            name="POST /fast_chat_v2"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Update conversation history if memory is enabled
                    if payload.get("memory_enabled") and data.get("conversation_history"):
                        self.conversation_history = data["conversation_history"]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)  # Weight: 2 - medium frequency
    def test_fast_chat_v2_stream_endpoint(self):
        """Test the /fast_chat_v2/stream endpoint"""
        query = random.choice(self.financial_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/fast_chat_v2/stream",
            json=payload,
            catch_response=True,
            name="POST /fast_chat_v2/stream"
        ) as response:
            if response.status_code == 200:
                # Read the stream content
                content = response.content.decode('utf-8')
                if content:
                    response.success()
                else:
                    response.failure("Empty stream response")
            else:
                response.failure(f"Status code: {response.status_code}")

class MixedUser(HttpUser):
    """Simulates a user that uses both chat and financial endpoints"""
    
    wait_time = between(1, 4)
    
    def on_start(self):
        """Initialize user session"""
        self.conversation_history = []
        self.chat_queries = [
            "What are the key financial metrics for JSE companies?",
            "Show me the revenue trends for major companies",
            "What about profit margins in the technology sector?"
        ]
        self.financial_queries = [
            "Show me MDS revenue for 2024",
            "Compare JBG and CPJ profit margins",
            "What about 2022?"
        ]
    
    @task(2)
    def test_chat_endpoint(self):
        """Test the /chat endpoint"""
        query = random.choice(self.chat_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "auto_load_documents": random.choice([True, False]),
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="POST /chat (mixed user)"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if payload.get("memory_enabled") and data.get("conversation_history"):
                        self.conversation_history = data["conversation_history"]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def test_fast_chat_v2_endpoint(self):
        """Test the /fast_chat_v2 endpoint"""
        query = random.choice(self.financial_queries)
        
        payload = {
            "query": query,
            "conversation_history": self.conversation_history,
            "memory_enabled": random.choice([True, False])
        }
        
        with self.client.post(
            "/fast_chat_v2",
            json=payload,
            catch_response=True,
            name="POST /fast_chat_v2 (mixed user)"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if payload.get("memory_enabled") and data.get("conversation_history"):
                        self.conversation_history = data["conversation_history"]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is starting"""
    print("🚀 Load test starting...")
    print(f"Target host: {environment.host}")
    print("Available endpoints:")
    print("  - POST /chat")
    print("  - POST /chat/stream")
    print("  - POST /fast_chat_v2")
    print("  - POST /fast_chat_v2/stream")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is stopping"""
    print("🏁 Load test completed!")
    print("Check the Locust web UI for detailed results and charts.")

