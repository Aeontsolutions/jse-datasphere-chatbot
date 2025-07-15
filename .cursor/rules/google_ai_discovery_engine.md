# Google AI Discovery Engine Integration Rules

## Authentication
- Always use `gcloud auth print-access-token` to get fresh access tokens
- Store tokens securely and refresh them as needed
- Never hardcode credentials in the code

## API Structure
### Search Endpoint
```python
POST https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/servingConfigs/{SERVING_CONFIG}:search
```

### Answer Endpoint
```python
POST https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/servingConfigs/{SERVING_CONFIG}:answer
```

## Request/Response Patterns

### Search Request
```python
{
    "query": query,
    "pageSize": 10,
    "queryExpansionSpec": {"condition": "AUTO"},
    "spellCorrectionSpec": {"mode": "AUTO"},
    "languageCode": "en-GB",
    "contentSearchSpec": {
        "extractiveContentSpec": {"maxExtractiveAnswerCount": 1}
    },
    "userInfo": {"timeZone": "Asia/Tokyo"},
    "session": f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/sessions/-"
}
```

### Answer Request
```python
{
    "query": {
        "text": original_query,  # Always include the original query text
        "queryId": query_id      # From search response
    },
    "session": session,          # From search response
    "relatedQuestionsSpec": {
        "enable": True
    },
    "answerGenerationSpec": {
        "ignoreAdversarialQuery": True,
        "ignoreNonAnswerSeekingQuery": False,
        "ignoreLowRelevantContent": True,
        "multimodalSpec": {},
        "includeCitations": True,
        "answerLanguageCode": "en",
        "modelSpec": {
            "modelVersion": "gemini-2.0-flash-001/answer_gen/v1"
        }
    }
}
```

## Best Practices

### Error Handling
- Always implement comprehensive error handling
- Log both request and response data for debugging
- Include raw response text in logs before parsing
- Handle HTTP errors gracefully with user-friendly messages

### Logging
```python
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('google_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### Session Management
- Extract `queryId` and `session` from `sessionInfo` in search response
- Maintain conversation history in Streamlit session state
- Clear session state when needed (e.g., new conversation)

### Response Processing
- Always check for required fields in responses
- Handle missing or malformed data gracefully
- Extract and display citations when available
- Format responses for better readability

## Common Issues and Solutions

### Missing queryId or session
- Check `sessionInfo` in search response
- Verify response structure matches API documentation
- Log complete response for debugging

### 400 Bad Request
- Verify all required fields are present
- Check data types and formats
- Include original query text in answer request
- Validate request structure against API spec

### Authentication Errors
- Ensure gcloud is properly configured
- Check token expiration
- Verify project permissions

## Testing
- Test with various query types
- Verify error handling
- Check response formatting
- Test session management
- Validate citation handling

## Security Considerations
- Never expose access tokens in client-side code
- Use environment variables for sensitive data
- Implement proper session management
- Follow least privilege principle for API access 