#!/bin/bash
# Startup script for ChatREL v4 Stack (Redis + Web + Celery)

echo "Starting ChatREL v4 Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker."
    exit 1
fi

# Start Redis
echo "Starting Redis..."
docker-compose up -d redis

# Wait for Redis
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
        echo "Redis is ready!"
        break
    fi
    sleep 1
done

# Start Web and Celery
echo "Starting Web and Celery services..."
docker-compose up -d web celery_worker celery_beat

echo "Stack started successfully!"
echo "Web UI: http://localhost:5000"
echo "Health Check: http://localhost:5000/health"
