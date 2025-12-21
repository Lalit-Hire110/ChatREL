#!/bin/bash
# Validation script for ChatREL v4 Stack

echo "Validating ChatREL v4 Stack..."

# Check Health Endpoint
echo "Checking /health endpoint..."
HEALTH=$(curl -s http://localhost:5000/health)
echo "Health Response: $HEALTH"

if echo "$HEALTH" | grep -q '"status": "ok"'; then
    echo "Health Check: PASS"
else
    echo "Health Check: FAIL"
    exit 1
fi

# Check Redis
echo "Checking Redis..."
if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
    echo "Redis Check: PASS"
else
    echo "Redis Check: FAIL"
    exit 1
fi

# Check Celery Worker
echo "Checking Celery Worker..."
if docker-compose logs celery_worker | grep -q "ready"; then
    echo "Celery Worker Check: PASS (based on logs)"
else
    echo "Celery Worker Check: WARNING (check logs manually)"
fi

echo "Validation Complete!"
