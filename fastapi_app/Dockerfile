# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory to project root
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_db \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./requirements.txt

# Copy companies.json file
COPY companies.json ./companies.json

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app using uvicorn with absolute import path
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]