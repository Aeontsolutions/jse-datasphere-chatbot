#!/usr/bin/env python3
"""
Simple smoke test to verify the ChromaDB refactor changes work correctly.
This test doesn't require external dependencies.
"""

import sys
import os

# Add the fastapi_app directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'fastapi_app'))

def test_query_collection_import():
    """Test that we can import the refactored query_collection function."""
    try:
        from fastapi_app.app.chroma_utils import query_collection
        print("✅ Successfully imported query_collection function")
        return True
    except ImportError as e:
        print(f"❌ Failed to import query_collection: {e}")
        return False

def test_function_signature():
    """Test that the function has the expected signature."""
    try:
        from fastapi_app.app.chroma_utils import query_collection
        import inspect
        
        sig = inspect.signature(query_collection)
        params = list(sig.parameters.keys())
        
        expected_params = ['collection', 'query', 'n_results', 'where']
        
        if params == expected_params:
            print("✅ Function signature is correct")
            return True
        else:
            print(f"❌ Function signature mismatch. Expected: {expected_params}, Got: {params}")
            return False
    except Exception as e:
        print(f"❌ Error checking function signature: {e}")
        return False

def test_main_api_imports():
    """Test that main API file can import successfully."""
    try:
        # This will test if our changes broke any imports
        from fastapi_app.app.main import app
        print("✅ Successfully imported FastAPI app")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False

def test_models_import():
    """Test that models can be imported."""
    try:
        from fastapi_app.app.models import ChromaQueryRequest, ChromaQueryResponse
        
        # Test that where parameter is now available
        import inspect
        sig = inspect.signature(ChromaQueryRequest)
        
        if 'where' in sig.parameters:
            print("✅ ChromaQueryRequest has 'where' parameter")
            return True
        else:
            print("❌ ChromaQueryRequest missing 'where' parameter")
            return False
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("🧪 Running ChromaDB refactor smoke tests...\n")
    
    tests = [
        test_query_collection_import,
        test_function_signature,
        test_models_import,
        test_main_api_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
        print()  # Add blank line between tests
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All smoke tests passed! The refactor appears to be working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the changes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)