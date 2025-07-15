#!/usr/bin/env python3
"""
Validation script to demonstrate the single-clause $and filter fix.

This script shows that the query_collection function correctly handles 
different filter scenarios without causing ChromaDB errors.
"""

import sys
import os
from unittest.mock import Mock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.chroma_utils import query_collection


def create_mock_collection():
    """Create a mock ChromaDB collection for testing."""
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Document 1 content", "Document 2 content"]],
        "metadatas": [[
            {"year": "2023", "company_name": "Test Co", "file_type": "financial"}, 
            {"year": "2022", "company_name": "Test Co", "file_type": "financial"}
        ]]
    }
    return mock_collection


def test_scenario(name, mock_companies, mock_doctype, expected_filter):
    """Test a specific filter scenario."""
    print(f"\n=== Testing {name} ===")
    
    # Mock the functions
    from unittest.mock import patch
    
    mock_collection = create_mock_collection()
    
    with patch('app.chroma_utils.get_companies_from_query', return_value=mock_companies), \
         patch('app.chroma_utils.get_doctype_from_query', return_value=mock_doctype):
        
        # Call the function
        try:
            result = query_collection(mock_collection, f"Test query for {name}", n_results=5)
            
            # Check what filter was used
            call_args = mock_collection.query.call_args
            actual_filter = call_args[1]['where']
            
            print(f"Expected filter: {expected_filter}")
            print(f"Actual filter:   {actual_filter}")
            
            if actual_filter == expected_filter:
                print("✅ PASS - Filter matches expected")
            else:
                print("❌ FAIL - Filter does not match expected")
                return False
                
            print(f"✅ Query executed successfully, returned {len(result[0])} results")
            return True
            
        except Exception as e:
            print(f"❌ FAIL - Exception occurred: {e}")
            return False


def main():
    """Run validation tests for different filter scenarios."""
    print("🔍 Validating ChromaDB Single-Clause $and Filter Fix")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Zero Filter Clauses",
            "companies": [],
            "doctype": ["unknown"],
            "expected": None
        },
        {
            "name": "Single Company Filter (Previously Failed)",
            "companies": ["Test Company"],
            "doctype": ["unknown"],
            "expected": {"company_name": {"$in": ["Test Company"]}}
        },
        {
            "name": "Single DocType Filter (Previously Failed)",
            "companies": [],
            "doctype": ["financial"],
            "expected": {"file_type": {"$in": ["financial"]}}
        },
        {
            "name": "Multiple Filters (Uses $and)",
            "companies": ["Test Company"],
            "doctype": ["financial"],
            "expected": {
                "$and": [
                    {"company_name": {"$in": ["Test Company"]}},
                    {"file_type": {"$in": ["financial"]}}
                ]
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        success = test_scenario(
            test_case["name"],
            test_case["companies"],
            test_case["doctype"],
            test_case["expected"]
        )
        results.append(success)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Single-clause $and filter fix is working correctly!")
        print("\nKey benefits:")
        print("- No more 'Expected where value for $and...' errors")
        print("- Single metadata filters work without $and wrapper")
        print("- Multiple filters correctly use $and")
        print("- Zero filters properly return None")
        return 0
    else:
        print("❌ Some tests failed - please check the implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())