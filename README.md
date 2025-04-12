# JSE Document Chat

A chat interface for querying and analyzing JSE (Johannesburg Stock Exchange) documents using AI.

## Features

- Natural language queries about JSE documents
- AI-powered document analysis and recommendations
- Conversation memory for contextual understanding
- View actual document content used for answers
- Document recommendations based on queries

## Prerequisites

- Python 3.10 or higher
- AWS credentials configured in environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`
  - `DOCUMENT_METADATA_S3_BUCKET`
- Google Cloud credentials:
  - `GOOGLE_APPLICATION_CREDENTIALS` pointing to your service account key file

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

## Running the Application

The application consists of two components that need to be run separately:

### 1. API Backend

The API backend provides the core functionality for document processing and AI interactions.

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### 2. Streamlit Frontend

The Streamlit frontend provides a user-friendly interface for interacting with the API.

```bash
# In a new terminal, start the Streamlit interface
streamlit run frontend.py
```

The frontend will be available at `http://localhost:8501`.

## Testing the Application

1. Start the API backend first:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. In a separate terminal, start the Streamlit frontend:
```bash
streamlit run frontend.py
```

3. Open your browser to `http://localhost:8501`

4. Use the chat interface to:
   - Ask questions about JSE documents
   - View the actual documents used for answers
   - See document recommendations
   - Configure settings in the sidebar

## Development

- The API backend is built with FastAPI
- The frontend is built with Streamlit
- AI capabilities are powered by Google's Vertex AI
- Document storage and retrieval uses AWS S3

## Troubleshooting

If you encounter issues:

1. Check that all required environment variables are set
2. Verify that both the API and frontend are running
3. Ensure you have the correct AWS and Google Cloud credentials
4. Check the logs in both terminal windows for error messages

## License

[Your License Here]
