# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_db

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure financial data files are present (these should be copied with the above COPY . . but being explicit)
# The financial_data.csv and metadata_for_fast_chat_v2.json files should be in the container
RUN if [ ! -f "financial_data.csv" ]; then echo "Warning: financial_data.csv not found - fast_chat_v2 endpoint may not work"; fi
RUN if [ ! -f "metadata_for_fast_chat_v2.json" ]; then echo "Warning: metadata_for_fast_chat_v2.json not found"; fi

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
