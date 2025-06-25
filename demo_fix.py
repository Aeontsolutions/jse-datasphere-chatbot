#!/usr/bin/env python3
"""
Demo script showing the multi-company embedding selection fix in action.

This script demonstrates how the improved semantic_document_selection function
now handles multi-company queries correctly.
"""

import json
from unittest.mock import patch, MagicMock

def demo_multi_company_fix():
    """Demonstrate the multi-company fix with before/after comparison."""
    
    print("🎯 Multi-Company Embedding Selection Fix Demo")
    print("=" * 60)
    
    # Sample metadata representing documents from different companies
    sample_metadata = [
        {
            "company": "Access Bank Plc",
            "filename": "access_annual_2022.txt", 
            "document_link": "s3://bucket/access_annual_2022.txt",
            "year": "2022",
            "document_type": "Annual Report"
        },
        {
            "company": "Access Bank Plc",
            "filename": "access_quarterly_q2_2022.txt",
            "document_link": "s3://bucket/access_quarterly_q2_2022.txt", 
            "year": "2022",
            "document_type": "Quarterly Report"
        },
        {
            "company": "NCB Financial Group Limited",
            "filename": "ncb_annual_2022.txt",
            "document_link": "s3://bucket/ncb_annual_2022.txt",
            "year": "2022", 
            "document_type": "Annual Report"
        },
        {
            "company": "NCB Financial Group Limited",
            "filename": "ncb_quarterly_q2_2022.txt",
            "document_link": "s3://bucket/ncb_quarterly_q2_2022.txt",
            "year": "2022",
            "document_type": "Quarterly Report"
        }
    ]
    
    # Import the improved function
    import sys
    import os
    sys.path.append('fastapi_app')
    from fastapi_app.app.utils import semantic_document_selection
    
    # Test query mentioning multiple companies
    test_query = "How did Access Bank compare to NCB in 2022?"
    
    print(f"📝 Test Query: '{test_query}'")
    print(f"📊 Available Documents: {len(sample_metadata)} from 2 companies")
    print()
    
    # Mock the dependencies to simulate the fix working
    with patch('fastapi_app.app.utils.get_companies_from_query') as mock_get_companies, \
         patch('fastapi_app.app.utils.GenerativeModel') as mock_model_class:
        
        # Setup: Mock company detection to find both companies
        mock_get_companies.return_value = ["Access Bank Plc", "NCB Financial Group Limited"]
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock LLM responses for each company (simulating separate searches)
        access_response = MagicMock()
        access_response.text = json.dumps([
            {
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "filename": "access_annual_2022.txt",
                "reason": "Access Bank annual report contains comprehensive 2022 financial data"
            },
            {
                "company": "Access Bank Plc", 
                "document_link": "s3://bucket/access_quarterly_q2_2022.txt",
                "filename": "access_quarterly_q2_2022.txt",
                "reason": "Q2 2022 quarterly results for additional context"
            }
        ])
        
        ncb_response = MagicMock()
        ncb_response.text = json.dumps([
            {
                "company": "NCB Financial Group Limited",
                "document_link": "s3://bucket/ncb_annual_2022.txt", 
                "filename": "ncb_annual_2022.txt",
                "reason": "NCB annual report for direct comparison with Access Bank"
            }
        ])
        
        # Set up the mock to return different responses for each company
        mock_model.generate_content.side_effect = [access_response, ncb_response]
        
        # Execute the improved function
        result = semantic_document_selection(test_query, sample_metadata)
        
        # Display results
        print("✅ AFTER FIX - Multi-Company Results:")
        print(f"   Companies Detected: {result['companies_mentioned']}")
        print(f"   Documents Selected: {len(result['documents_to_load'])}")
        print()
        
        for i, doc in enumerate(result['documents_to_load'], 1):
            print(f"   {i}. {doc['filename']} ({doc['company']})")
            print(f"      Reason: {doc['reason']}")
            print()
        
        print("🔧 How the Fix Works:")
        print("   1. Extracts both 'Access Bank Plc' and 'NCB Financial Group Limited' from query")
        print("   2. Runs separate document search for each company")
        print("   3. Gets top 3 documents per company (instead of top 5 total)")
        print("   4. Merges and deduplicates results")
        print("   5. Returns documents from BOTH companies for comparison")
        print()
        
        print("🚀 Impact:")
        print("   ❌ Before: Only Access Bank documents (first company found)")
        print("   ✅ After: Documents from both Access Bank AND NCB")
        print("   📈 Enables proper multi-company comparisons!")
        print()
        
        # Verify the fix worked correctly
        companies_in_results = set(doc['company'] for doc in result['documents_to_load'])
        assert "Access Bank Plc" in companies_in_results
        assert "NCB Financial Group Limited" in companies_in_results
        
        print("✅ Demo completed successfully - Multi-company fix is working!")
        
        return True

if __name__ == "__main__":
    demo_multi_company_fix()