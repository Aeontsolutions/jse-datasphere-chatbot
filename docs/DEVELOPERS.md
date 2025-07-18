# Developer Guide

This guide provides comprehensive technical information for developers working on the JSE DataSphere Chatbot project.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Structure & Organization](#code-structure--organization)
4. [Core Components](#core-components)
5. [API Design & Patterns](#api-design--patterns)
6. [Data Flow & Processing](#data-flow--processing)
7. [Testing Strategy](#testing-strategy)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Deployment & DevOps](#deployment--devops)
11. [Troubleshooting & Debugging](#troubleshooting--debugging)
12. [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

### System Architecture

The JSE DataSphere Chatbot is built as a microservices architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Frontend  │  REST API Clients  │  Mobile Apps       │
└──────────────────────┴─────────────────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application (Port 8000)                                │
│  - Request Routing & Validation                                 │
│  - Authentication & Authorization                               │
│  - Rate Limiting & Caching                                      │
│  - Logging & Monitoring                                         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Business Logic Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Document Processing  │  Financial Analytics  │  AI Services    │
│  - PDF Extraction     │  - BigQuery Queries   │  - Gemini AI    │
│  - Text Processing    │  - Data Aggregation   │  - Embeddings   │
│  - Metadata Mgmt      │  - Financial Metrics  │  - LLM Parsing  │
└──────────────────────┴─────────────────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  Vector Database  │  Document Storage  │  Financial Data       │
│  - ChromaDB       │  - AWS S3          │  - Google BigQuery    │
│  - Embeddings     │  - PDF Files       │  - Structured Data    │
│  - Semantic Search│  - Metadata        │  - Time Series        │
└──────────────────────┴─────────────────────┴─────────────────────┘
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend Framework** | FastAPI | Latest | High-performance async API |
| **Python Runtime** | Python | 3.11+ | Core application language |
| **Vector Database** | ChromaDB | Latest | Semantic search & embeddings |
| **AI Models** | Google Gemini | 2.0 Flash | Natural language processing |
| **Document Storage** | AWS S3 | - | PDF document storage |
| **Financial Data** | Google BigQuery | - | Structured financial data |
| **Containerization** | Docker | Latest | Application packaging |
| **Orchestration** | AWS Copilot | Latest | Cloud deployment |
| **Frontend** | Streamlit | Latest | Web interface (optional) |

## Development Environment Setup

### Prerequisites

1. **Python Environment**
   ```bash
   # Install Python 3.11+
   python --version  # Should be 3.11 or higher
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Docker Setup**
   ```bash
   # Install Docker Desktop
   docker --version
   docker-compose --version
   ```

3. **Cloud CLI Tools**
   ```bash
   # AWS CLI
   aws --version
   aws configure  # Set up credentials
   
   # Google Cloud CLI (optional)
   gcloud --version
   gcloud auth login
   ```

### Local Development Setup

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd jse-datasphere-chatbot
   
   # Install dependencies
   cd fastapi_app
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Configure required variables
   nano .env
   ```

3. **Database Setup**
   ```bash
   # Start ChromaDB (Docker)
   docker-compose up -d chroma
   
   # Or local ChromaDB
   pip install chromadb[server]
   chroma run --host localhost --port 8000
   ```

4. **Verify Setup**
   ```bash
   # Test API
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Check health
   curl http://localhost:8000/health
   ```

### IDE Configuration

#### VS Code Setup
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm Setup
1. Configure Python interpreter to use virtual environment
2. Enable type checking
3. Configure pytest as test runner
4. Set up code formatting with Black

## Code Structure & Organization

### Project Layout
```
fastapi_app/
├── app/                          # Core application package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application & endpoints
│   ├── models.py                # Pydantic data models
│   ├── utils.py                 # Core utilities & AI integration
│   ├── chroma_utils.py          # Vector database operations
│   ├── financial_utils.py       # BigQuery financial data manager
│   ├── streaming_chat.py        # Streaming response handling
│   └── progress_tracker.py      # Progress tracking utilities
├── tests/                       # Test suite
│   ├── test_api.py             # API endpoint tests
│   ├── test_financial_utils.py # Financial data tests
│   ├── test_chroma_utils.py    # Vector DB tests
│   ├── test_cache_optimization.py # Caching tests
│   └── test_streaming.py       # Streaming tests
├── copilot/                     # AWS Copilot deployment configs
│   ├── environments/           # Environment-specific configs
│   ├── api/                    # API service config
│   └── chroma/                 # ChromaDB service config
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Local development services
└── .env.example               # Environment template
```

### Code Organization Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Services are injected via FastAPI dependencies
3. **Type Safety**: Comprehensive type hints throughout
4. **Error Handling**: Consistent error handling patterns
5. **Logging**: Structured logging for observability

## Core Components

### 1. FastAPI Application (`app/main.py`)

The main application orchestrates all components and provides the REST API interface.

**Key Features:**
- Application lifecycle management
- Middleware configuration (CORS, logging, etc.)
- Dependency injection setup
- Health checks and monitoring
- Request/response logging

**Important Patterns:**
```python
# Dependency injection
def get_s3_client():
    return app.state.s3_client

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Logging implementation
```

### 2. Data Models (`app/models.py`)

Pydantic models define the API contract and ensure data validation.

**Key Models:**
- `ChatRequest` / `ChatResponse`: Core chat functionality
- `FinancialDataRequest` / `FinancialDataResponse`: Financial queries
- `StreamingChatRequest`: Streaming chat support
- `ChromaQueryRequest` / `ChromaQueryResponse`: Vector search

**Validation Patterns:**
```python
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query/question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Previous conversation history"
    )
    auto_load_documents: bool = Field(default=True)
    memory_enabled: bool = Field(default=True)
```

### 3. Vector Database Operations (`app/chroma_utils.py`)

Handles all ChromaDB interactions for semantic search and document storage.

**Key Functions:**
- `init_chroma_client()`: Database connection setup
- `get_or_create_collection()`: Collection management
- `query_collection()`: Semantic search operations
- `add_documents()`: Document ingestion

**Usage Patterns:**
```python
# Initialize client
client = init_chroma_client()

# Get collection with embeddings
collection = get_or_create_collection(client, "documents")

# Query documents
results = query_collection(collection, query="financial results", n_results=5)
```

### 4. Financial Data Manager (`app/financial_utils.py`)

Manages BigQuery integration for structured financial data queries.

**Key Features:**
- Natural language query parsing
- BigQuery SQL generation
- Data aggregation and formatting
- Conversation context management

**Architecture:**
```python
class FinancialDataManager:
    def __init__(self):
        self._initialize_bigquery_client()
        self._initialize_ai_model()
        self.load_metadata_from_bigquery()
    
    def parse_user_query(self, query: str, conversation_history=None):
        # LLM-based query parsing
        pass
    
    def query_data(self, filters: FinancialDataFilters):
        # BigQuery data retrieval
        pass
```

### 5. AI Integration (`app/utils.py`)

Handles Google AI services integration and caching optimization.

**Key Components:**
- Google Gemini model initialization
- Context caching for performance
- Document selection algorithms
- Response generation

**Caching Strategy:**
```python
# Metadata caching for performance
_metadata_cache = None
_cache_expiry = None
_cache_hash = None

def create_metadata_cache(metadata):
    # Create Gemini context cache
    pass

def get_cached_model(metadata):
    # Retrieve cached model for reuse
    pass
```

## API Design & Patterns

### RESTful Endpoint Design

The API follows RESTful principles with consistent patterns:

```python
# Standard response format
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, ...):
    try:
        # Process request
        result = await process_chat(request)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Streaming API Pattern

For long-running operations, the API provides streaming responses:

```python
@app.post("/chat/stream")
async def chat_stream(request: StreamingChatRequest, ...):
    tracker = await process_streaming_chat(request, ...)
    return StreamingResponse(
        tracker.stream_updates(),
        media_type="text/event-stream"
    )
```

### Error Handling Strategy

Consistent error handling across all endpoints:

```python
# Standard error responses
class APIError(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIError(error="Internal server error").dict()
    )
```

## Data Flow & Processing

### Document Processing Pipeline

```
User Query → Document Selection → Vector Search → Content Retrieval → AI Response
     │              │                │                │              │
     ▼              ▼                ▼                ▼              ▼
Query Analysis   Metadata      ChromaDB Query    S3 Download   Gemini AI
                 Collection    (Embeddings)      (PDF Text)    Generation
```

### Financial Data Flow

```
Natural Language Query → LLM Parsing → BigQuery SQL → Data Retrieval → Response Generation
         │                    │              │              │              │
         ▼                    ▼              ▼              ▼              ▼
User Input              Structured      Parameterized   Financial      Formatted
                        Filters         SELECT Query    Records        Response
```

### Caching Strategy

Multi-level caching for optimal performance:

1. **Metadata Cache**: Gemini context caching for document metadata
2. **Query Cache**: Cached BigQuery results for common queries
3. **Embedding Cache**: Cached document embeddings in ChromaDB

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **API Tests**: Endpoint functionality testing
4. **Performance Tests**: Load and latency testing

### Test Structure

```python
# Example test structure
class TestFinancialDataManager:
    def setup_method(self):
        """Setup test fixtures"""
        self.manager = FinancialDataManager()
    
    def test_parse_user_query(self):
        """Test natural language query parsing"""
        query = "Show me NCB revenue for 2023"
        filters = self.manager.parse_user_query(query)
        assert filters.companies == ["NCB"]
        assert filters.years == ["2023"]
    
    def test_bigquery_integration(self):
        """Test BigQuery data retrieval"""
        # Mock BigQuery client
        # Test data retrieval
        pass
```

### Test Execution

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_financial_utils.py -v
python -m pytest tests/test_api_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### Mocking Strategy

```python
# Example mocking for external services
@patch('app.financial_utils.bigquery.Client')
def test_bigquery_connection(mock_client):
    mock_client.return_value.query.return_value.result.return_value = []
    manager = FinancialDataManager()
    # Test with mocked BigQuery
```

## Performance Optimization

### Document Selection Optimization

**Before**: LLM-based selection (~20s)
**After**: Embedding-based selection (~300ms)

```python
def semantic_document_selection(query, metadata, conversation_history=None, meta_collection=None):
    if meta_collection:
        # Fast embedding-based selection
        return query_meta_collection(meta_collection, query)
    else:
        # Fallback to LLM-based selection
        return semantic_document_selection_llm_fallback(query, metadata, conversation_history)
```

### Caching Optimization

**Context Caching**: 85% latency reduction for LLM fallback scenarios

```python
def get_cached_model(metadata):
    global _metadata_cache, _cache_expiry, _cache_hash
    
    current_hash = get_metadata_hash(metadata)
    if (_metadata_cache and _cache_expiry and 
        datetime.utcnow() < _cache_expiry and 
        _cache_hash == current_hash):
        return _metadata_cache
    
    return create_metadata_cache(metadata)
```

### Database Optimization

**BigQuery Best Practices:**
- Use parameterized queries
- Implement proper filtering
- Leverage BigQuery's columnar storage
- Use appropriate data types

```python
def query_data(self, filters: FinancialDataFilters) -> List[FinancialDataRecord]:
    # Build efficient query with proper filters
    query = f"""
        SELECT Company, Symbol, Year, standard_item, item_value, unit_multiplier
        FROM `{self.project_id}.{self.dataset}.{self.table}`
        WHERE 1=1
    """
    
    # Add filters dynamically
    if filters.companies:
        query += f" AND Company IN UNNEST(@companies)"
    
    # Execute with parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("companies", "STRING", filters.companies)
        ]
    )
```

## Security Considerations

### Input Validation

All inputs are validated using Pydantic models:

```python
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        max_items=50  # Prevent memory exhaustion
    )
```

### SQL Injection Prevention

All BigQuery queries use parameterized queries:

```python
# Safe: Parameterized query
query = "SELECT * FROM table WHERE company = @company"
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("company", "STRING", company_name)
    ]
)

# Never: String concatenation
# query = f"SELECT * FROM table WHERE company = '{company_name}'"  # UNSAFE
```

### Credential Management

Credentials are managed securely:

```python
# Environment-based credentials
service_account_info = os.getenv("GCP_SERVICE_ACCOUNT_INFO")
if service_account_info:
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(service_account_info)
    )

# File-based credentials (development)
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and os.path.exists(credentials_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
```

### API Security

- CORS configuration for web clients
- Rate limiting (implemented via middleware)
- Request logging for audit trails
- Error message sanitization

## Deployment & DevOps

### Docker Configuration

**Multi-stage Dockerfile:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### AWS Copilot Deployment

**Service Configuration:**
```yaml
# copilot/api/manifest.yml
name: api
type: Load Balanced Web Service

http:
  path: '/'
  healthcheck: 
    path: '/health'
    interval: 30s
    timeout: 10s
    retries: 3

image:
  build: Dockerfile
  port: 8000

cpu: 512
memory: 1024
count: 1
```

**Environment-Specific Configurations:**
```yaml
environments:
  dev:
    count: 1
    cpu: 512
    memory: 1024
  staging:
    count: 1
    cpu: 512
    memory: 1024
  prod:
    count: 2
    cpu: 1024
    memory: 2048
    deployment:
      rolling: 'recreate'
```

### CI/CD Pipeline

**GitHub Actions Example:**
```yaml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r fastapi_app/requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          cd fastapi_app
          python -m pytest tests/ --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to staging
        run: |
          cd fastapi_app/copilot
          copilot deploy --name api --env staging
```

## Troubleshooting & Debugging

### Common Issues

#### 1. ChromaDB Connection Issues
```bash
# Check ChromaDB status
curl http://localhost:8001/api/v1/heartbeat

# Check logs
docker-compose logs chroma

# Restart service
docker-compose restart chroma
```

#### 2. BigQuery Connection Issues
```bash
# Verify credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $GCP_SERVICE_ACCOUNT_INFO

# Test connection
python -c "
from google.cloud import bigquery
client = bigquery.Client()
print('BigQuery connection successful')
"
```

#### 3. S3 Access Issues
```bash
# Test S3 connectivity
aws s3 ls s3://your-bucket-name

# Check AWS credentials
aws sts get-caller-identity

# Verify bucket permissions
aws s3api get-bucket-policy --bucket your-bucket-name
```

### Debugging Tools

#### 1. Application Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use structured logging
logger = logging.getLogger(__name__)
logger.info("Processing request", extra={
    "request_id": request_id,
    "user_id": user_id,
    "operation": "chat"
})
```

#### 2. Performance Monitoring
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f}s")
```

#### 3. Health Checks
```python
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "s3": check_s3_connection(),
            "bigquery": check_bigquery_connection(),
            "chromadb": check_chromadb_connection()
        }
    }
    return health_status
```

### Performance Debugging

#### 1. Query Performance
```python
# BigQuery query performance
def query_with_timing(self, query, parameters=None):
    start_time = time.time()
    try:
        result = self.bq_client.query(query, job_config=parameters)
        rows = list(result.result())
        duration = time.time() - start_time
        logger.info(f"BigQuery query completed in {duration:.2f}s")
        return rows
    except Exception as e:
        logger.error(f"BigQuery query failed after {time.time() - start_time:.2f}s: {e}")
        raise
```

#### 2. Memory Usage
```python
import psutil
import os

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
```

## Contributing Guidelines

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/jse-datasphere-chatbot.git
   cd jse-datasphere-chatbot
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development**
   - Follow coding standards
   - Write comprehensive tests
   - Update documentation

4. **Testing**
   ```bash
   # Run test suite
   python -m pytest tests/ -v
   
   # Check code coverage
   python -m pytest tests/ --cov=app --cov-report=html
   
   # Run linting
   flake8 app/ tests/
   black app/ tests/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Provide clear description
   - Include test results
   - Reference related issues

### Code Standards

#### 1. Python Style Guide
- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints for all functions

#### 2. Documentation Standards
```python
def process_chat_request(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request and generate a response.
    
    Args:
        request: The chat request containing query and context
        
    Returns:
        ChatResponse: The generated response with documents and metadata
        
    Raises:
        ValueError: If request validation fails
        ConnectionError: If external services are unavailable
    """
    pass
```

#### 3. Testing Standards
- Minimum 80% code coverage
- Unit tests for all public functions
- Integration tests for API endpoints
- Performance tests for critical paths

#### 4. Git Commit Standards
Use conventional commit messages:
```
feat: add new feature
fix: bug fix
docs: documentation changes
style: formatting changes
refactor: code refactoring
test: adding or updating tests
chore: maintenance tasks
```

### Review Process

1. **Code Review Checklist**
   - [ ] Code follows style guidelines
   - [ ] Tests are comprehensive and passing
   - [ ] Documentation is updated
   - [ ] Security considerations addressed
   - [ ] Performance impact assessed

2. **Review Guidelines**
   - Be constructive and specific
   - Focus on code quality and maintainability
   - Consider security implications
   - Verify test coverage

3. **Approval Requirements**
   - At least one maintainer approval
   - All CI checks passing
   - No security vulnerabilities
   - Adequate test coverage

### Release Process

1. **Version Management**
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Update version in `app/main.py`
   - Create release notes

2. **Deployment Pipeline**
   - Deploy to staging environment
   - Run integration tests
   - Deploy to production
   - Monitor for issues

3. **Post-Release**
   - Monitor application metrics
   - Address any issues promptly
   - Update documentation if needed

---

This developer guide provides comprehensive information for contributing to the JSE DataSphere Chatbot project. For additional questions or clarifications, please refer to the project documentation or create an issue in the repository. 