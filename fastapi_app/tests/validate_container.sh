#!/bin/bash

# This script validates the Docker container for the FastAPI API

echo "Building Docker container..."
docker-compose build

echo "Starting Docker container..."
docker-compose up -d

echo "Waiting for container to start..."
sleep 10

echo "Testing API endpoints..."
python test_api.py

echo "Stopping Docker container..."
docker-compose down

echo "Validation complete!"
