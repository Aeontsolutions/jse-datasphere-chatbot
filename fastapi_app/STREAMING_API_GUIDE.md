# Streaming API Guide

This guide explains how to use the new streaming endpoints that provide real-time progress updates while processing chat requests.

## Overview

The streaming API uses **Server-Sent Events (SSE)** to send real-time progress updates to the frontend while processing chat requests. This gives users visibility into what's happening during potentially long-running operations.

## Available Endpoints

### 1. `/chat/stream` - Traditional Chat with Streaming
- **Method**: POST
- **Content-Type**: application/json
- **Response**: text/event-stream

Provides the same functionality as `/chat` but with real-time progress updates.

### 2. `/fast_chat/stream` - Vector DB Chat with Streaming  
- **Method**: POST
- **Content-Type**: application/json
- **Response**: text/event-stream

Provides the same functionality as `/fast_chat` but with real-time progress updates.

## Request Format

Both endpoints accept the same request body:

```json
{
  "query": "Your question here",
  "auto_load_documents": true,
  "memory_enabled": true,
  "conversation_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

## Response Format

The streaming response uses Server-Sent Events with the following event types:

### Progress Events
```
event: progress
data: {
  "step": "doc_loading",
  "message": "Loading relevant documents from S3...",
  "progress": 45.0,
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "documents_loaded": 3,
    "companies": ["Company A", "Company B"]
  }
}
```

### Result Event (Final Response)
```
event: result
data: {
  "response": "The AI-generated response text",
  "documents_loaded": ["doc1.pdf", "doc2.pdf"],
  "document_selection_message": "Selected 2 relevant documents",
  "conversation_history": [...]
}
```

### Error Event
```
event: error
data: {
  "error": "Error message describing what went wrong"
}
```

### Heartbeat Event
```
event: heartbeat
data: {}
```

## Progress Steps

### Traditional Chat (`/chat/stream`)
1. **start** (5%) - Starting chat processing
2. **doc_loading** (20-60%) - Loading documents from S3
3. **ai_generation** (80%) - Generating AI response
4. **finalizing** (95%) - Finalizing response
5. **complete** (100%) - Processing complete

### Fast Chat (`/fast_chat/stream`)
1. **start** (5%) - Starting chat processing
2. **query_prep** (10%) - Preparing search query
3. **doc_selection** (25-35%) - Selecting relevant documents
4. **vector_search** (50-65%) - Searching vector database
5. **ai_generation** (80%) - Generating AI response
6. **finalizing** (95%) - Finalizing response
7. **complete** (100%) - Processing complete

## Frontend Implementation

### JavaScript Example

```javascript
function startStreamingChat(query) {
    const requestData = {
        query: query,
        auto_load_documents: true,
        memory_enabled: true,
        conversation_history: conversationHistory
    };
    
    fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        function processStream() {
            return reader.read().then(({ done, value }) => {
                if (done) return;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                let currentEvent = '';
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7);
                    } else if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        handleStreamEvent(currentEvent, data);
                    }
                }
                
                return processStream();
            });
        }
        
        return processStream();
    });
}

function handleStreamEvent(event, data) {
    switch (event) {
        case 'progress':
            updateProgress(data.message, data.progress);
            break;
        case 'result':
            showFinalResult(data);
            break;
        case 'error':
            showError(data.error);
            break;
    }
}
```

### React Example with EventSource

```javascript
import { useEffect, useState } from 'react';

function useStreamingChat() {
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    
    const startChat = async (query) => {
        setProgress(0);
        setMessage('');
        setResult(null);
        setError(null);
        
        try {
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                let currentEvent = '';
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7);
                    } else if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        
                        switch (currentEvent) {
                            case 'progress':
                                setProgress(data.progress);
                                setMessage(data.message);
                                break;
                            case 'result':
                                setResult(data);
                                break;
                            case 'error':
                                setError(data.error);
                                break;
                        }
                    }
                }
            }
        } catch (err) {
            setError(err.message);
        }
    };
    
    return { progress, message, result, error, startChat };
}
```

## Testing

### 1. HTML Test Client
Use the provided `streaming_test_client.html` file to test the streaming functionality:

```bash
# Open the HTML file in your browser
open fastapi_app/streaming_test_client.html
```

### 2. Streamlit Test Client
Use the updated Streamlit test client with the "Streaming Chat" mode:

```bash
# Run the Streamlit app
cd fastapi_app
streamlit run test_client.py
```

### 3. curl Example
```bash
# Test the streaming endpoint with curl
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main financial highlights?",
    "auto_load_documents": true,
    "memory_enabled": true
  }' \
  --no-buffer
```

## Error Handling

The streaming API handles errors gracefully:

1. **Connection Errors**: If the connection is lost, the client should retry
2. **Processing Errors**: Sent as `error` events with descriptive messages
3. **Timeout**: The stream includes heartbeat events to keep connections alive

## Best Practices

1. **Always handle all event types** (progress, result, error, heartbeat)
2. **Implement connection retry logic** for production use
3. **Show progress indicators** to improve user experience
4. **Handle errors gracefully** and provide fallback options
5. **Use heartbeat events** to detect connection issues

## Deployment Considerations

### CORS Settings
Ensure your CORS settings allow streaming:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Reverse Proxy
If using nginx or similar, ensure streaming is properly configured:

```nginx
location /chat/stream {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
}
```

### Load Balancers
Be aware that some load balancers may buffer responses. Configure them to support streaming for these endpoints.

## Monitoring

Monitor streaming endpoints for:
- Connection duration
- Error rates
- Progress step timing
- Memory usage during long-running operations

The streaming implementation provides detailed logging for debugging and monitoring purposes. 