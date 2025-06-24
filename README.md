# JSE Document Chat

A sophisticated chat interface for querying and analyzing Jamaica Stock Exchange (JSE) documents using AI and vector-based semantic search.

## Features

- Natural language queries about JSE documents using advanced AI models
- Vector-based semantic search powered by ChromaDB
- AI-powered document analysis and recommendations
- Conversation memory for contextual understanding
- View actual document content used for answers
- Document recommendations based on queries
- PDF processing and text extraction capabilities
- Integration with Google Cloud's Vertex AI and AWS S3

## Prerequisites

- Python 3.10 or higher
- AWS credentials configured in environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`
  - `DOCUMENT_METADATA_S3_BUCKET`
- Google Cloud credentials:
  - `GOOGLE_APPLICATION_CREDENTIALS` pointing to your service account key file
- Docker and Docker Compose (for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jse-datasphere-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Initial Setup

### Populate Document Metadata Collection

For optimal performance with the new embedding-based document selection, populate the metadata collection:

```bash
# Run the population script to set up metadata collection
python scripts/populate_doc_meta.py
```

This script:
- Reads document metadata from your S3 bucket
- Creates embeddings for each document's metadata
- Populates the `doc_meta` ChromaDB collection
- Enables sub-second document selection (~300ms vs ~20s)

The metadata collection contains document information like:
- Company name
- Document type (financial, non-financial)
- Reporting period
- Filename

**Note:** This step is optional but highly recommended for production deployments to achieve the best performance.

## Running the Application

The application consists of two components that need to be run separately:

### 1. API Backend

The API backend provides the core functionality for document processing, vector search, and AI interactions.

#### Local Development
```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

#### Key API Endpoints

**Semantic Document Selection** (New - High Performance)
- `POST /fast_chat` - Enhanced chat endpoint with embedding-based document selection
  - Uses vector embeddings for sub-second document selection (~300ms vs ~20s)
  - Falls back to LLM-based selection if embedding search fails
  - Significantly reduces overall latency

**Metadata Collection Management** (New)
- `POST /chroma/meta/update` - Add/update document metadata for embedding search
- `POST /chroma/meta/query` - Query document metadata collection directly

**Document Management**
- `POST /chroma/update` - Add documents to the vector database
- `POST /chroma/query` - Query documents using semantic search
- `POST /chat` - Original chat endpoint with full document loading

For detailed API documentation, visit `http://localhost:8000/docs` when the server is running.

#### Docker Deployment
1. Ensure your `.env` file is properly configured with all required environment variables.

2. Place your Google Cloud service account key file as `service-account.json` in the project root.

3. Build and start the container:
```bash
docker-compose up -d
```

4. To stop the container:
```bash
docker-compose down
```

The API will be available at `http://localhost:8000` when running in Docker.

### 2. Streamlit Frontend

The Streamlit frontend provides a user-friendly interface for interacting with the API.

```bash
# In a new terminal, start the Streamlit interface
streamlit run frontend.py
```

The frontend will be available at `http://localhost:8501`.

## Project Structure

- `api.py` - FastAPI backend with document processing and AI integration
- `frontend.py` - Streamlit-based user interface
- `app.py` - Core application logic and utilities
- `utils/` - Utility functions and helper modules
- `chroma/` - Vector database storage for semantic search
- `pdfs/` - Directory for storing and processing PDF documents

## Development

- The API backend is built with FastAPI
- The frontend is built with Streamlit
- AI capabilities are powered by Google's Vertex AI and Google Generative AI
- Document storage and retrieval uses AWS S3
- Vector search is implemented using ChromaDB
- PDF processing uses PyPDF2
- Natural Language Processing uses spaCy 3.8.3

## Troubleshooting

If you encounter issues:

1. Check that all required environment variables are set in your `.env` file
2. Verify that both the API and frontend are running
3. Ensure you have the correct AWS and Google Cloud credentials
4. Check the logs in both terminal windows for error messages
5. For Docker issues:
   - Check Docker logs: `docker-compose logs api`
   - Verify environment variables are set correctly
   - Ensure the service account file is present and valid
6. For ChromaDB issues:
   - Verify the `chroma/` directory exists and has proper permissions
   - Check if the vector database is properly initialized
7. For PDF processing issues:
   - Ensure PDF files are not corrupted
   - Check if PyPDF2 can properly read the files

## License
