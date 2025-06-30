# Fast Chat V2 - Financial Data Query Endpoint

## Overview

The `/fast_chat_v2` endpoint provides natural language querying capabilities for financial data. It uses AI to parse user queries, extract relevant filters, and return structured financial information with natural language responses.

## Features

- **Natural Language Processing**: Ask questions in plain English
- **Conversation Memory**: Supports follow-up questions and context awareness
- **Smart Filtering**: Automatically identifies companies, symbols, years, and metrics
- **Rich Responses**: Returns both AI-generated explanations and structured data
- **Data Validation**: Provides warnings and suggestions for data availability

## API Endpoints

### POST `/fast_chat_v2`

Query financial data using natural language.

**Request Body:**
```json
{
    "query": "Show me MDS revenue for 2024",
    "conversation_history": [
        {"role": "user", "content": "Previous query"},
        {"role": "assistant", "content": "Previous response"}
    ],
    "memory_enabled": true
}
```

**Response:**
```json
{
    "response": "AI-generated natural language response",
    "data_found": true,
    "record_count": 25,
    "filters_used": {
        "companies": ["Medical Disposables & Supplies Limited"],
        "symbols": ["MDS"],
        "years": ["2024"],
        "standard_items": ["revenue"],
        "interpretation": "User wants revenue data for MDS in 2024",
        "is_follow_up": false
    },
    "data_preview": [
        {
            "company": "Medical Disposables & Supplies Limited",
            "symbol": "MDS",
            "year": "2024",
            "standard_item": "revenue",
            "item_value": 3709.0,
            "unit_multiplier": 1000000,
            "formatted_value": "3.71B"
        }
    ],
    "conversation_history": [...],
    "warnings": ["Optional warnings about data availability"],
    "suggestions": ["Optional suggestions for better queries"]
}
```

### GET `/financial/metadata`

Get metadata about available financial data.

**Response:**
```json
{
    "status": "success",
    "metadata": {
        "companies": ["Medical Disposables & Supplies Limited", ...],
        "symbols": ["MDS", "SOS", "JBG", ...],
        "years": ["2019", "2020", "2021", "2022", "2023", "2024"],
        "standard_items": ["revenue", "net_profit", "total_assets", ...],
        "total_records": 2101,
        "associations": {...}
    }
}
```

### GET `/health`

Check the health status of the API including financial data availability.

**Response:**
```json
{
    "status": "healthy",
    "s3_client": "available",
    "metadata": "available",
    "financial_data": {
        "status": "available",
        "records": 2101
    }
}
```

## Example Queries

### Basic Queries
- `"Show me revenue for all companies in 2024"`
- `"What is MDS revenue for 2023?"`
- `"Compare JBG and SOS profit margins"`

### Symbol-based Queries
- `"SOS financial data for 2024"`
- `"MDS vs JBG revenue comparison"`
- `"Show me BGA gross profit"`

### Follow-up Queries (with conversation history)
- `"What about 2022?"` (after discussing a company)
- `"Show me their assets instead"` (changes metric but keeps company/year)
- `"How about gross profit?"` (changes metric)

### Time-based Queries
- `"Show revenue trends from 2020 to 2024"`
- `"Latest financial data for MDS"`
- `"Historical profit data"`

### Ratio/Metrics Queries
- `"Show me debt to equity ratios"`
- `"Compare ROE for all companies"`
- `"EPS data for 2024"`

## Setup Instructions

1. **Prerequisites:**
   - Python 3.8+
   - FastAPI
   - pandas
   - google-generativeai (for AI parsing)

2. **Required Files:**
   - `financial_data.csv` - Your financial data file
   - `.env` file with `GEMINI_API_KEY` (optional, but recommended for better parsing)

3. **File Structure:**
   ```
   fastapi_app/
   ├── financial_data.csv
   ├── app/
   │   ├── main.py
   │   ├── models.py
   │   ├── financial_utils.py
   │   └── ...
   └── ...
   ```

4. **Start the API:**
   ```bash
   cd fastapi_app
   uvicorn app.main:app --reload
   ```

## Data Format

The endpoint expects a CSV file with the following columns:
- `Company` - Full company name
- `Symbol` - Stock trading symbol
- `Year` - Year of the data
- `standard_item` - Standardized metric name (e.g., "revenue", "net_profit")
- `item_value` - The numeric value
- `unit_multiplier` - Multiplier for the value (1, 1000000, 1000000000)
- `item_type` - Type of item ("line item", "ratio", etc.)

Optional columns:
- `calculated_value` - Pre-calculated actual value
- `confidence` - Confidence level of the data
- `Item` - Original item name

## Testing

Use the provided test script:

```bash
python test_fast_chat_v2.py
```

This will run comprehensive tests of the endpoint with various query types.

## Error Handling

The endpoint provides helpful error messages for:
- Missing financial data files
- Invalid query formats
- Data availability issues
- AI parsing failures (falls back to manual parsing)

## Performance Notes

- The endpoint uses caching for metadata to improve response times
- Large datasets are automatically paginated in responses
- AI parsing provides intelligent fallbacks for better reliability
- Query results are limited to 50 records in the preview for performance

## Limitations

- Requires `financial_data.csv` to be present and properly formatted
- AI parsing requires `GEMINI_API_KEY` for optimal performance (has fallback)
- Conversation history is limited to the last 20 exchanges
- Data preview is limited to 50 records per response 