# JSE DataSphere Chatbot - Development Guide

This guide covers the development setup, workflows, and best practices for the JSE DataSphere Chatbot.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Code Quality Tools](#code-quality-tools)
- [Docker Development](#docker-development)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Python**: 3.10 or 3.11 (3.12 not yet supported)
- **pip**: >= 21.0 (for pyproject.toml support)
- **Docker**: Latest version (for containerized development)
- **Git**: For version control

### Cloud Service Access
- AWS account with S3 access
- Google Cloud project with BigQuery and Vertex AI enabled
- Service account credentials with appropriate permissions

## Installation

### Development Environment Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd jse-datasphere-chatbot/fastapi_app
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

**For Development (Recommended):**
```bash
# Install all dependencies including development tools
pip install -e ".[dev]"
```

This installs:
- All production dependencies
- Development tools (pytest, black, ruff, mypy)
- Pre-commit hooks support

**For Production Only:**
```bash
# Install only production dependencies
pip install .
```

#### 4. Install Pre-commit Hooks
```bash
# Install pre-commit hooks
cd ..  # Go to project root
pre-commit install

# Test pre-commit hooks (optional)
pre-commit run --all-files
```

#### 5. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
DOCUMENT_METADATA_S3_BUCKET=your-bucket

# Google Cloud Configuration
GOOGLE_API_KEY=your_gemini_api_key
GCP_SERVICE_ACCOUNT_INFO={"type":"service_account",...}
GCP_PROJECT_ID=your-project-id
BIGQUERY_DATASET=your_dataset
BIGQUERY_TABLE=your_table

# API Keys
SUMMARIZER_API_KEY=your_key
CHATBOT_API_KEY=your_key
```

## Development Workflow

### Starting the Development Server

#### Local Development
```bash
# From fastapi_app directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables hot reloading for development.

#### Docker Development
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Project Structure
```
fastapi_app/
├── app/                         # Core application code
│   ├── main.py                 # FastAPI app and endpoints
│   ├── models.py               # Pydantic models
│   ├── utils.py                # Core utilities
│   ├── financial_utils.py      # BigQuery financial data
│   ├── streaming_chat.py       # Streaming responses
│   └── progress_tracker.py     # Progress tracking
├── tests/                       # Test suite
│   ├── test_api.py            # API endpoint tests
│   ├── test_financial_utils.py # Financial data tests
│   └── test_streaming.py      # Streaming tests
├── pyproject.toml              # Project dependencies and config
├── requirements.txt            # Frozen dependencies (Docker)
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Local development setup
└── DEVELOPMENT.md             # This file
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py

# Run specific test function
pytest tests/test_api.py::test_health_endpoint

# Run tests in parallel (faster)
pytest -n auto
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov

# Generate HTML coverage report
pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Async Tests

The project uses `pytest-asyncio` for testing async functions:
```python
import pytest

@pytest.mark.asyncio
async def test_async_endpoint():
    result = await some_async_function()
    assert result is not None
```

## Code Quality Tools

All tools are configured in `pyproject.toml` for consistency.

### Black (Code Formatting)

```bash
# Format all Python files
black .

# Check formatting without changes
black --check .

# Format specific file
black app/main.py
```

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.10+

### Ruff (Linting)

```bash
# Lint all files
ruff check .

# Lint with auto-fix
ruff check --fix .

# Lint specific file
ruff check app/main.py
```

Configuration in `pyproject.toml`:
- Enabled rules: pycodestyle, pyflakes, isort, flake8-bugbear
- Line length: 100 characters

### MyPy (Type Checking)

```bash
# Type check all files
mypy app/

# Type check specific file
mypy app/main.py
```

Configuration in `pyproject.toml`:
- Python version: 3.10
- Ignore missing imports: true

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Run manually on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate
```

Hooks include:
- Trailing whitespace removal
- YAML/JSON syntax checking
- Black formatting
- Ruff linting
- Secret detection
- Security scanning

## Docker Development

### Building the Image

```bash
# Build the image
docker-compose build

# Build without cache
docker-compose build --no-cache
```

### Running Services

```bash
# Start services in background
docker-compose up -d

# Start services with logs
docker-compose up

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Container Management

```bash
# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec api bash

# View container stats
docker stats

# Inspect container
docker-compose exec api env
```

### Health Checks

```bash
# Check container health
docker-compose ps

# Test API health endpoint
curl http://localhost:8000/health
```

## Dependency Management

### Adding New Dependencies

1. Add to `pyproject.toml` under `dependencies`:
```toml
[project]
dependencies = [
    "new-package>=1.0.0",
]
```

2. For development dependencies:
```toml
[project.optional-dependencies]
dev = [
    "new-dev-tool>=2.0.0",
]
```

3. Update requirements.txt:
```bash
pip install -e ".[dev]"
pip freeze > requirements.txt.new
# Review and merge into requirements.txt
```

4. Test the changes:
```bash
# Local testing
pip install -e ".[dev]"
pytest

# Docker testing
docker-compose build
docker-compose up -d
```

### Updating Dependencies

```bash
# Update all dependencies
pip install --upgrade -e ".[dev]"

# Update specific package
pip install --upgrade package-name

# Regenerate requirements.txt
pip freeze > requirements.txt
```

## API Development

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are NCB financial results?"}'

# Financial data endpoint
curl -X POST "http://localhost:8000/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show NCB revenue for 2023"}'
```

### Adding New Endpoints

1. Define Pydantic models in `app/models.py`
2. Add endpoint in `app/main.py`
3. Write tests in `tests/`
4. Update API documentation
5. Run tests and linting

Example:
```python
# app/models.py
class NewRequest(BaseModel):
    field: str

# app/main.py
@app.post("/new-endpoint")
async def new_endpoint(request: NewRequest):
    return {"result": process(request.field)}

# tests/test_api.py
def test_new_endpoint():
    response = client.post("/new-endpoint", json={"field": "test"})
    assert response.status_code == 200
```

## Performance Optimization

### Profiling

```bash
# Profile API endpoints
python -m cProfile -o profile.stats app/main.py

# Analyze profile
python -m pstats profile.stats
```

### Monitoring Cache

```bash
# Check cache status
curl http://localhost:8000/cache/status

# Refresh cache
curl -X POST http://localhost:8000/cache/refresh
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in editable mode
pip install -e ".[dev]"

# Verify PYTHONPATH
echo $PYTHONPATH

# Check installed packages
pip list | grep jse-datasphere
```

#### Google Cloud Authentication
```bash
# Verify service account
echo $GCP_SERVICE_ACCOUNT_INFO

# Test BigQuery connection
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('OK')"
```

#### AWS S3 Access
```bash
# Test AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://$DOCUMENT_METADATA_S3_BUCKET
```

#### Docker Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```

## Best Practices

### Code Quality
- Write type hints for all functions
- Add docstrings for public APIs
- Keep functions small and focused
- Use meaningful variable names
- Follow PEP 8 style guide

### Testing
- Write tests for new features
- Test edge cases and error handling
- Use fixtures for common setup
- Mock external services
- Aim for >80% coverage

### Git Workflow
- Create feature branches from `main`
- Write descriptive commit messages
- Keep commits atomic and focused
- Run tests before pushing
- Request code reviews

### Security
- Never commit secrets or credentials
- Use environment variables for config
- Validate all user inputs
- Use parameterized queries
- Keep dependencies updated

## Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

### Project Documentation
- [Main README](../README.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE_FAST_CHAT_V2.md)
- [Streaming Guide](docs/STREAMING_API_GUIDE.md)

### Support
- Create GitHub issues for bugs
- Use GitHub Discussions for questions
- Check existing documentation first

---

**Happy Coding!** 🚀
