1# Fast Chat V2 Endpoint (`/fast_chat_v2`)

## Overview

The `/fast_chat_v2` endpoint provides a fast, conversational interface for querying financial data using natural language. It leverages Google BigQuery for data storage and retrieval, and uses Gemini AI for intelligent query parsing and context handling. This endpoint is designed for speed, robustness, and ease of extension.

---

## How It Works

1. **User Query**: The user sends a POST request with a natural language question (e.g., "What was KREMI's revenue in 2024?").
2. **AI Parsing**: The Gemini AI model parses the query, extracting filters such as companies, symbols, years, and financial metrics. It also considers conversation history for follow-up questions.
3. **Data Availability Check**: The system checks if the requested data exists in BigQuery metadata, providing warnings or suggestions if data is missing.
4. **BigQuery Retrieval**: If data is available, a parameterized BigQuery query fetches the relevant records.
5. **Response Formatting**: Results are formatted into a user-friendly, conversational response, optionally using Gemini AI for natural language output.

---

## Request Structure

**Endpoint:**
```
POST /fast_chat_v2
```

**Request Body Example:**
```json
{
  "query": "What was KREMI's revenue in 2024?",
  "conversation_history": [
    {"role": "user", "content": "Show me KREMI's financials."}
  ],
  "memory_enabled": true
}
```

- `query`: The user's natural language question.
- `conversation_history`: (Optional) List of previous exchanges for context.
- `memory_enabled`: (Optional, default true) Whether to use conversation memory.

---

## Response Structure

**Success Example:**
```json
{
  "response": "Caribbean Cream Limited (KREMI) reported revenue of 2.64B in 2024.",
  "data_found": true,
  "record_count": 1,
  "filters_used": { ... },
  "data_preview": [ ... ],
  "conversation_history": [ ... ],
  "warnings": [],
  "suggestions": []
}
```
- `response`: AI-generated answer.
- `data_found`: Whether any data matched the query.
- `record_count`: Number of records found.
- `filters_used`: The parsed filters (companies, symbols, years, etc.).
- `data_preview`: Preview of returned records.
- `warnings`: Any issues (e.g., missing data).
- `suggestions`: Alternative queries or available data.

---

## Key Components

### 1. **Gemini AI Query Parsing**
- Uses Gemini to extract structured filters from user queries.
- Handles follow-up questions by referencing conversation history.
- Falls back to manual parsing if AI is unavailable.

### 2. **BigQuery Integration**
- Metadata (companies, symbols, years, items) is loaded from BigQuery at startup.
- Data queries are parameterized to prevent SQL injection and ensure safety.
- Handles BigQuery Row quirks (e.g., fields named 'items' that may collide with methods).

### 3. **Error Handling & Debugging**
- Extensive logging at INFO level for all major steps (parsing, validation, querying).
- Robust handling of BigQuery Row objects to avoid method/field name collisions.
- Pydantic models enforce strict validation of request and response data.
- Warnings and suggestions are provided if requested data is missing.

### 4. **Recent Lessons & Fixes**
- **BigQuery Row 'items' field**: Special care is taken to extract the 'items' field from BigQuery Row objects, avoiding accidental use of the `.items()` method.
- **Model Alignment**: The response model (`FinancialDataRecord`) is kept in sync with the fields provided by the query logic to avoid validation errors.
- **Debug Logging**: Logs are used to trace the flow and catch issues early, especially with data extraction and model instantiation.

---

## Troubleshooting

- **Missing Data**: If the response contains warnings about missing data, check the metadata associations in BigQuery.
- **Validation Errors**: Ensure that all required fields in the Pydantic models are provided by the query logic.
- **BigQuery Row Issues**: If you see errors about 'method is not iterable', review how fields are extracted from BigQuery Row objects (see code comments for robust extraction patterns).
- **Debugging**: Use the logs to trace the flow of data and identify where issues occur. Logs at the INFO level will show the type and value of key variables at each step.

---

## Extending the Endpoint

- To add new filters or metrics, update the metadata loading and parsing logic.
- To support new data sources, extend the query logic to handle additional tables or APIs.
- To improve AI parsing, refine the Gemini prompt or add more context from conversation history.

---

## Contact
For questions or contributions, please contact the project maintainers or open an issue in the repository. 