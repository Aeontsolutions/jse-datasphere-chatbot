#!/usr/bin/env python3
"""
Test script for the new /fast_chat_v2 endpoint
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust if your API runs on a different port
ENDPOINT = "/fast_chat_v2"

def test_endpoint(query: str, conversation_history=None, memory_enabled=True):
    """Test the fast_chat_v2 endpoint with a query"""
    
    url = f"{API_BASE_URL}{ENDPOINT}"
    
    payload = {
        "query": query,
        "memory_enabled": memory_enabled
    }
    
    if conversation_history:
        payload["conversation_history"] = conversation_history
    
    print(f"\n🚀 Testing query: '{query}'")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success!")
            print(f"Data Found: {data.get('data_found', False)}")
            print(f"Record Count: {data.get('record_count', 0)}")
            print(f"AI Response: {data.get('response', '')[:200]}...")
            
            if data.get('warnings'):
                print(f"⚠️  Warnings: {data['warnings']}")
            
            if data.get('suggestions'):
                print(f"💡 Suggestions: {data['suggestions']}")
            
            # Show filters used
            filters = data.get('filters_used', {})
            print(f"\n🔍 Filters Applied:")
            print(f"  Companies: {filters.get('companies', [])}")
            print(f"  Symbols: {filters.get('symbols', [])}")
            print(f"  Years: {filters.get('years', [])}")
            print(f"  Items: {filters.get('standard_items', [])}")
            print(f"  Follow-up: {filters.get('is_follow_up', False)}")
            
            # Show data preview
            if data.get('data_preview'):
                print(f"\n📈 Data Preview (first 3 records):")
                for i, record in enumerate(data['data_preview'][:3]):
                    print(f"  {i+1}. {record['company']} ({record['symbol']}) - {record['year']}")
                    print(f"     {record['standard_item']}: {record['formatted_value']}")
            
            return data.get('conversation_history', [])
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {API_BASE_URL}")
        print("Make sure the FastAPI server is running with: uvicorn app.main:app --reload")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None

def test_health_check():
    """Test the health endpoint to check if financial data is available"""
    url = f"{API_BASE_URL}/health"
    
    print(f"\n🏥 Health Check: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"Financial Data Status: {data.get('financial_data', {}).get('status', 'unknown')}")
            print(f"Financial Records: {data.get('financial_data', {}).get('records', 0)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_financial_metadata():
    """Test the financial metadata endpoint"""
    url = f"{API_BASE_URL}/financial/metadata"
    
    print(f"\n📊 Financial Metadata: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            metadata = data.get('metadata', {})
            
            print(f"✅ Metadata available")
            print(f"Total Records: {metadata.get('total_records', 0)}")
            print(f"Companies: {len(metadata.get('companies', []))}")
            print(f"Symbols: {len(metadata.get('symbols', []))}")
            print(f"Years: {metadata.get('years', [])}")
            print(f"Metrics: {len(metadata.get('standard_items', []))}")
            
            return True
        else:
            print(f"❌ Metadata check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metadata check error: {str(e)}")
        return False

def main():
    """Run comprehensive tests of the fast_chat_v2 endpoint"""
    
    print("🧪 Testing Fast Chat V2 Endpoint")
    print("=" * 50)
    
    # First check if the service is healthy
    if not test_health_check():
        print("\n❌ Service not healthy. Ensure the FastAPI server is running and financial_data.csv exists.")
        return
    
    # Check metadata
    if not test_financial_metadata():
        print("\n⚠️  Financial metadata not available. Some tests may fail.")
    
    # Test cases
    test_queries = [
        # Basic queries
        "Show me revenue for all companies in 2024",
        "What is MDS revenue for 2023?",
        "Compare JBG and CPJ profit margins",
        
        # Follow-up queries (would work in a real conversation)
        "What about 2022?",
        "Show me their assets instead",
        
        # Complex queries
        "Show me SOS financial data for the last 3 years",
        "Compare revenue and net profit for all companies",
        
        # Edge cases
        "Revenue for XYZ company",  # Non-existent company
        "Show me data for 1999",   # Year that might not exist
    ]
    
    conversation_history = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_queries)}")
        
        # For follow-up queries, use conversation history
        if query in ["What about 2022?", "Show me their assets instead"] and conversation_history:
            result = test_endpoint(query, conversation_history=conversation_history, memory_enabled=True)
        else:
            result = test_endpoint(query, memory_enabled=True)
        
        if result:
            conversation_history = result
        
        # Small delay between requests
        import time
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("🎉 Testing completed!")
    print("Check the logs above for any issues.")

if __name__ == "__main__":
    main() 