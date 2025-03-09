# JSE Datasphere Chatbot

A Streamlit application that allows users to chat with PDF documents stored in AWS S3 buckets using Google's Vertex AI Gemini models.

## Features

- PDF document processing from uploads or S3 storage
- Semantic document selection based on user queries
- Conversation memory to maintain context across queries
- Integration with Google Vertex AI Gemini models
- Docker support for easy deployment

## Prerequisites

- AWS account with S3 access
- Google Cloud account with Vertex AI access
- Docker and Docker Compose (for containerized deployment)

## Environment Setup

The application requires the following environment variables to be set in a `.env` file:

```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region
DOCUMENT_METADATA_S3_BUCKET=jse-metadata-bucket
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
```

An example file `.env.example` is provided as a template.

## Service Account

A Google Cloud service account JSON file is required for Vertex AI authentication. Save this file as `service-account.json` in the project root directory.

## Metadata Structure

The application expects a metadata.json file in the S3 bucket that provides information about available documents. The structure should be:

```json
{
  "company1": [
    {
      "filename": "document1.pdf",
      "document_type": "Annual Report",
      "period": "2023",
      "document_link": "s3://jse-renamed-docs/organized/path/to/document1.pdf"
    },
    ...
  ],
  "company2": [
    ...
  ]
}
```

## Running Locally

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Run the application:
   ```
   streamlit run app.py
   ```

## Docker Deployment

1. Build and start the container:
   ```
   docker-compose up -d
   ```

2. To stop the container:
   ```
   docker-compose down
   ```

## Usage

1. Access the application at http://localhost:8501
2. Select a document source (Upload, Manual Selection, or Automatic)
3. Ask questions about the documents
4. The application will automatically select relevant documents based on your queries

## File Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.dockerignore` - Files to exclude from Docker image
- `.env.example` - Example environment variables
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose configuration
- `create_docker_files.bat` - Windows batch script to generate Docker files

## Dependencies

- streamlit - Web application framework
- boto3 - AWS SDK for Python
- python-dotenv - Environment variable management
- google-auth, google-cloud-aiplatform - Google Cloud authentication and Vertex AI
- PyPDF2 - PDF processing
- vertexai - Google Vertex AI client
- pycryptodome - Cryptographic library

## Security Notes

- Never commit sensitive files like `.env` or `service-account.json` to version control
- Use environment variables for all sensitive credentials
- Keep AWS and Google Cloud credentials secure
