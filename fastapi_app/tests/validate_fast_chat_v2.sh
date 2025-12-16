#!/bin/bash

# Validation script for fast_chat_v2 endpoint
# This script validates that all dependencies for the financial data endpoint are available

echo "🔍 Validating fast_chat_v2 endpoint dependencies..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Make sure to run this from the app directory."
    exit 1
fi

echo "✅ Application directory confirmed"

# Check for required Python packages
echo "📦 Checking Python dependencies..."

python3 -c "import pandas; print('✅ pandas available')" || echo "❌ pandas not available"
python3 -c "import fastapi; print('✅ fastapi available')" || echo "❌ fastapi not available"
python3 -c "import google.generativeai; print('✅ google.generativeai available')" || echo "❌ google.generativeai not available"

# Check for required data files
echo "📊 Checking data files..."

if [ -f "financial_data.csv" ]; then
    lines=$(wc -l < financial_data.csv)
    echo "✅ financial_data.csv found ($lines lines)"
else
    echo "❌ financial_data.csv not found - fast_chat_v2 endpoint will not work"
fi

if [ -f "metadata_for_fast_chat_v2.json" ]; then
    echo "✅ metadata_for_fast_chat_v2.json found"
else
    echo "❌ metadata_for_fast_chat_v2.json not found"
fi

if [ -f "companies.json" ]; then
    echo "✅ companies.json found"
else
    echo "❌ companies.json not found"
fi

# Check environment variables
echo "🔧 Checking environment variables..."

if [ -n "$GOOGLE_API_KEY" ] || [ -n "$GCP_SERVICE_ACCOUNT_INFO" ]; then
    echo "✅ Google AI credentials configured"
else
    echo "⚠️  Google AI credentials not found - fast_chat_v2 AI parsing may not work"
fi

if [ -n "$CHROMA_HOST" ]; then
    echo "✅ ChromaDB host configured: $CHROMA_HOST"
else
    echo "⚠️  CHROMA_HOST not configured"
fi

# Test basic imports for the financial utils
echo "🧪 Testing financial utilities..."

python3 -c "
try:
    import sys
    import os
    # Add the current directory to Python path so we can import from app/
    sys.path.insert(0, os.getcwd())
    from app.financial_utils import FinancialDataManager
    print('✅ FinancialDataManager import successful')
except Exception as e:
    print(f'❌ FinancialDataManager import failed: {e}')

try:
    # Also test the basic data loading
    sys.path.insert(0, os.getcwd())
    from app.financial_utils import FinancialDataManager
    manager = FinancialDataManager('financial_data.csv')
    if manager.load_data():
        print('✅ Financial data loading test successful')
    else:
        print('⚠️  Financial data loading test failed')
except Exception as e:
    print(f'⚠️  Financial data loading test failed: {e}')
"

echo "🎉 Validation complete!"
echo ""
echo "📝 Notes:"
echo "   - If you see any ❌ errors, the fast_chat_v2 endpoint may not work properly"
echo "   - ⚠️  warnings indicate features that may be limited but endpoint should still work"
echo "   - Make sure your .env file contains the necessary API keys"
echo ""
