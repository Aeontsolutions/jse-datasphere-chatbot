# Document Chat API Documentation

## Overview

This API provides document chat functionality using Gemini AI. It allows users to ask questions about documents stored in AWS S3 and receive AI-generated responses based on the document content.

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Environment Setup](#environment-setup)
3. [Request/Response Models](#requestresponse-models)
4. [Deployment Instructions](#deployment-instructions)
   - [Local Development](#local-development)
   - [Docker Deployment](#docker-deployment)
   - [AWS ECS Deployment](#aws-ecs-deployment)
5. [Usage Examples](#usage-examples)

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API and its dependencies.

**Response Example:**

```json
{
  "status": "healthy",
  "s3_client": "available",
  "metadata": "available"
}
```

### Chat

```
POST /chat
```

Chat with documents using Gemini AI.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | User query/question |
| conversation_history | array | No | Previous conversation history as a list of role-content pairs |
| auto_load_documents | boolean | No | Whether to automatically load relevant documents (default: true) |
| memory_enabled | boolean | No | Whether to use conversation memory (default: true) |

**Response Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| response | string | AI response to the query |
| documents_loaded | array | List of documents that were loaded to answer the query |
| document_selection_message | string | Message about document selection process |
| conversation_history | array | Updated conversation history |

## Environment Setup

The API requires the following environment variables:

| Variable | Description |
|----------|-------------|
| AWS_ACCESS_KEY_ID | AWS access key ID |
| AWS_SECRET_ACCESS_KEY | AWS secret access key |
| AWS_DEFAULT_REGION | AWS region |
| DOCUMENT_METADATA_S3_BUCKET | S3 bucket for document metadata |
| GCP_SERVICE_ACCOUNT_INFO | GCP service account info (JSON format as a string) |

Create a `.env` file based on the provided `.env.example` file.

## Request/Response Models

### ChatRequest

```json
{
  "query": "What is the revenue for Company X in 2023?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Tell me about Company X"
    },
    {
      "role": "assistant",
      "content": "Company X is a technology company founded in..."
    }
  ],
  "auto_load_documents": true,
  "memory_enabled": true
}
```

### ChatResponse

```json
{
  "response": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year.",
  "documents_loaded": [
    "Company_X_Annual_Report_2023.pdf"
  ],
  "document_selection_message": "Semantically selected 1 documents based on your query:\n• Company_X_Annual_Report_2023.pdf - Contains financial information for Company X for the year 2023",
  "conversation_history": [
    {
      "role": "user",
      "content": "Tell me about Company X"
    },
    {
      "role": "assistant",
      "content": "Company X is a technology company founded in..."
    },
    {
      "role": "user",
      "content": "What is the revenue for Company X in 2023?"
    },
    {
      "role": "assistant",
      "content": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year."
    }
  ]
}
```

## Deployment Instructions

### Local Development

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the API:
   ```
   python main.py
   ```
5. Access the API at http://localhost:8000

### Docker Deployment

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Build and run the Docker container:
   ```
   docker-compose up --build
   ```
4. Access the API at http://localhost:8000

### AWS ECS Deployment

#### Prerequisites

- AWS CLI installed and configured
- Docker installed
- ECR repository created

#### Steps

1. Build the Docker image:
   ```
   docker build -t document-chat-api .
   ```

2. Tag the image for ECR:
   ```
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag document-chat-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/document-chat-api:latest
   ```

3. Push the image to ECR:
   ```
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/document-chat-api:latest
   ```

4. Create an ECS cluster (if not already created):
   ```
   aws ecs create-cluster --cluster-name document-chat-cluster
   ```

5. Create a task definition:
   ```
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   ```

   Example `task-definition.json`:
   ```json
   {
     "family": "document-chat-task",
     "networkMode": "awsvpc",
     "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "document-chat-container",
         "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/document-chat-api:latest",
         "essential": true,
         "portMappings": [
           {
             "containerPort": 8000,
             "hostPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "AWS_DEFAULT_REGION",
             "value": "<region>"
           }
         ],
         "secrets": [
           {
             "name": "AWS_ACCESS_KEY_ID",
             "valueFrom": "arn:aws:ssm:<region>:<account-id>:parameter/document-chat/AWS_ACCESS_KEY_ID"
           },
           {
             "name": "AWS_SECRET_ACCESS_KEY",
             "valueFrom": "arn:aws:ssm:<region>:<account-id>:parameter/document-chat/AWS_SECRET_ACCESS_KEY"
           },
           {
             "name": "DOCUMENT_METADATA_S3_BUCKET",
             "valueFrom": "arn:aws:ssm:<region>:<account-id>:parameter/document-chat/DOCUMENT_METADATA_S3_BUCKET"
           },
           {
             "name": "GCP_SERVICE_ACCOUNT_INFO",
             "valueFrom": "arn:aws:ssm:<region>:<account-id>:parameter/document-chat/GCP_SERVICE_ACCOUNT_INFO"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/document-chat",
             "awslogs-region": "<region>",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ],
     "requiresCompatibilities": [
       "FARGATE"
     ],
     "cpu": "1024",
     "memory": "2048"
   }
   ```

6. Create a service:
   ```
   aws ecs create-service --cluster document-chat-cluster --service-name document-chat-service --task-definition document-chat-task --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<security-group-id>],assignPublicIp=ENABLED}"
   ```

7. Set up an Application Load Balancer (ALB) to route traffic to the service.

## Usage Examples

### Example 1: Simple Query

**Request:**
```json
{
  "query": "What is the revenue for Company X in 2023?"
}
```

**Response:**
```json
{
  "response": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year.",
  "documents_loaded": [
    "Company_X_Annual_Report_2023.pdf"
  ],
  "document_selection_message": "Semantically selected 1 documents based on your query:\n• Company_X_Annual_Report_2023.pdf - Contains financial information for Company X for the year 2023",
  "conversation_history": [
    {
      "role": "user",
      "content": "What is the revenue for Company X in 2023?"
    },
    {
      "role": "assistant",
      "content": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year."
    }
  ]
}
```

### Example 2: Query with Conversation History

**Request:**
```json
{
  "query": "How does that compare to the previous year?",
  "conversation_history": [
    {
      "role": "user",
      "content": "What is the revenue for Company X in 2023?"
    },
    {
      "role": "assistant",
      "content": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year."
    }
  ]
}
```

**Response:**
```json
{
  "response": "In 2022, Company X reported a revenue of $9.13 billion. The 2023 revenue of $10.5 billion represents a 15% increase from 2022.",
  "documents_loaded": [
    "Company_X_Annual_Report_2023.pdf",
    "Company_X_Annual_Report_2022.pdf"
  ],
  "document_selection_message": "Semantically selected 2 documents based on your query:\n• Company_X_Annual_Report_2023.pdf - Contains current year financial information\n• Company_X_Annual_Report_2022.pdf - Contains previous year financial information for comparison",
  "conversation_history": [
    {
      "role": "user",
      "content": "What is the revenue for Company X in 2023?"
    },
    {
      "role": "assistant",
      "content": "According to the 2023 annual report, Company X reported a revenue of $10.5 billion, which represents a 15% increase from the previous year."
    },
    {
      "role": "user",
      "content": "How does that compare to the previous year?"
    },
    {
      "role": "assistant",
      "content": "In 2022, Company X reported a revenue of $9.13 billion. The 2023 revenue of $10.5 billion represents a 15% increase from 2022."
    }
  ]
}
```
