# JSE DataSphere Chatbot

A sophisticated AI-powered chatbot for querying and analyzing Jamaica Stock Exchange (JSE) documents and financial data. The system combines natural language processing, vector-based semantic search, and financial data analytics to provide intelligent responses about JSE companies and their documents.

## 🚀 Features

### Core Capabilities
- **Natural Language Queries**: Ask questions about JSE documents using conversational language
- **Direct Document Loading**: Intelligent document retrieval from S3 storage
- **Financial Data Analytics**: BigQuery integration for structured financial data queries
- **AI-Powered Analysis**: Google Gemini models for document analysis and recommendations
- **Conversation Memory**: Contextual understanding across multiple interactions
- **Document Recommendations**: Smart suggestions based on query content

### Advanced Features
- **Streaming Responses**: Real-time progress updates during processing
- **Multi-Modal Document Support**: PDF processing and text extraction
- **Performance Optimization**: Caching and embedding-based document selection
- **Cloud Integration**: AWS S3 for document storage, Google Cloud for AI services
- **Production Ready**: Docker containerization and AWS Copilot deployment

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │
│   (Streamlit)   │◄──►│   Backend       │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Google Cloud  │    │   AWS S3        │
                       │   (BigQuery &   │    │   Document      │
                       │    Vertex AI)   │    │   Storage       │
                       └─────────────────┘    └─────────────────┘
```

### Key Technologies
- **Backend**: FastAPI (Python 3.11)
- **AI Models**: Google Gemini 2.0 Flash
- **Data Storage**: AWS S3 (documents), Google BigQuery (financial data)
- **Deployment**: Docker + AWS Copilot
- **Frontend**: Streamlit (optional)

## 📋 Prerequisites

### Required Software
- Python 3.11 or higher
- Docker and Docker Compose
- AWS CLI (for deployment)
- Google Cloud CLI (optional)

### Cloud Services
- **AWS Account** with S3 access
- **Google Cloud Project** with BigQuery and Vertex AI enabled
- **Service Account** with appropriate permissions

### Environment Variables
See `.env.example` for complete list. Key variables:
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
```

## 🛠️ Installation & Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd jse-datasphere-chatbot
```

### 2. Environment Configuration
```bash
# Copy environment template
cp fastapi_app/.env.example fastapi_app/.env

# Edit with your credentials
nano fastapi_app/.env
```

### 3. Local Development Setup

#### Option A: Docker (Recommended)
```bash
cd fastapi_app
docker-compose up -d
```

#### Option B: Local Python
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd fastapi_app
pip install -r requirements.txt

# Start services
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs
```

## 🚀 Quick Start

### 1. Start the Application
```bash
# Using Docker
cd fastapi_app
docker-compose up -d

# Or locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test Basic Functionality
```bash
# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest financial results for NCB?"}'

# Test financial data endpoint
curl -X POST "http://localhost:8000/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me NCB revenue for 2023"}'
```

### 3. Access Web Interface
```bash
# Start Streamlit frontend (optional)
streamlit run frontend.py
# Open http://localhost:8501
```

## 📚 API Endpoints

### Core Chat Endpoints
- `POST /chat` - Traditional chat with full document loading
- `POST /chat/stream` - Streaming chat with real-time progress updates
- `POST /fast_chat_v2` - Financial data queries with BigQuery integration

### Financial Data
- `POST /fast_chat_v2` - Natural language financial data queries
- `GET /financial/metadata` - Get available financial data metadata

### System Management
- `GET /health` - System health check
- `GET /cache/status` - Cache status and performance metrics
- `POST /cache/refresh` - Force refresh metadata cache

## 🧪 Testing

### Run Test Suite
```bash
cd fastapi_app
python -m pytest tests/ -v
```

### Key Test Categories
- **API Integration**: `test_api_integration.py`
- **Financial Data**: `test_financial_utils.py`
- **Caching**: `test_cache_optimization.py`
- **Streaming**: `test_streaming.py`

### Performance Testing
```bash
# Test document selection performance
python tests/test_meta_collection.py

# Test financial data queries
python tests/test_financial_utils.py
```

## 🚀 Deployment

### AWS Copilot Deployment
```bash
cd fastapi_app/copilot

# Deploy to development environment
copilot deploy --name api --env dev

# Deploy to production
copilot deploy --name api --env prod
```

### Docker Deployment
```bash
cd fastapi_app
docker-compose -f docker-compose.yml up -d
```

### Environment-Specific Configurations
- **Development**: Single instance, minimal resources
- **Staging**: Single instance, moderate resources
- **Production**: Multiple instances, high availability

## 📊 Monitoring & Logging

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Check cache status
curl http://localhost:8000/cache/status
```

### Logging
- **Application Logs**: Structured logging with request tracking
- **Performance Metrics**: Latency tracking and cache hit rates
- **Error Tracking**: Comprehensive error logging with context

### Key Metrics
- Document selection latency (~300ms vs ~20s traditional)
- Cache hit rates (85% latency reduction)
- API response times
- Error rates and types

## 🔧 Development

### Project Structure
```
fastapi_app/
├── app/                    # Core application code
│   ├── main.py            # FastAPI application and endpoints
│   ├── models.py          # Pydantic data models
│   ├── utils.py           # Core utilities and AI integration
│   ├── financial_utils.py # BigQuery financial data manager
│   ├── streaming_chat.py  # Streaming response handling
│   └── progress_tracker.py # Progress tracking utilities
├── tests/                 # Test suite
├── copilot/              # AWS Copilot deployment configs
├── docs/                 # Documentation
└── requirements.txt      # Python dependencies
```

### Development Workflow
1. **Feature Development**: Create feature branch from `main`
2. **Testing**: Write tests for new functionality
3. **Code Review**: Submit pull request with comprehensive tests
4. **Deployment**: Deploy to staging for validation
5. **Production**: Deploy to production after validation

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings for all public APIs
- **Testing**: Minimum 80% test coverage
- **Logging**: Structured logging for all major operations
- **Security**: Parameterized queries, input validation

## 🐛 Troubleshooting

### Common Issues

#### BigQuery Connection Issues
```bash
# Verify credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $GCP_SERVICE_ACCOUNT_INFO

# Test BigQuery connection
python -c "from google.cloud import bigquery; print('BigQuery OK')"
```

#### S3 Access Issues
```bash
# Test S3 connectivity
aws s3 ls s3://your-bucket-name

# Check AWS credentials
aws sts get-caller-identity
```

### Performance Optimization
- **Document Selection**: Use `/fast_chat_v2` for financial data queries
- **Caching**: Monitor cache hit rates and refresh when needed
- **BigQuery**: Use appropriate query filters to reduce data transfer

## 📖 Documentation

- **[API Documentation](fastapi_app/docs/API_DOCUMENTATION.md)** - Complete API reference
- **[Deployment Guide](fastapi_app/docs/DEPLOYMENT_GUIDE_FAST_CHAT_V2.md)** - Deployment instructions
- **[Streaming Guide](fastapi_app/docs/STREAMING_API_GUIDE.md)** - Streaming API usage
- **[Copilot Guide](fastapi_app/docs/COPILOT_DEPLOYMENT_GUIDE.md)** - AWS Copilot deployment

## 🤝 Contributing

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run the test suite: `python -m pytest tests/ -v`
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for API changes
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check the docs folder for detailed guides
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

### Contact
- **Project Maintainers**: [List maintainers]
- **Technical Support**: [Support contact information]

---

**Note**: This is a production system handling sensitive financial data. Always follow security best practices and ensure proper access controls are in place.
