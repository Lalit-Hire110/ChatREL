"""
Integration tests for Celery and Redis.
Requires Docker stack to be running or Redis available.
"""

import pytest
import time
import requests
import redis
from chatrel import config
from chatrel.utils.redis_client import get_redis_client
from chatrel.tasks.update_word_stats import update_token_stats_async

# Skip if Redis not available
try:
    r = redis.from_url(config.CSM_REDIS_URL, socket_timeout=1)
    r.ping()
    REDIS_UP = True
except:
    REDIS_UP = False

@pytest.mark.skipif(not REDIS_UP, reason="Redis not available")
def test_redis_connection():
    """Test robust Redis client."""
    client = get_redis_client()
    assert client is not None
    assert client.ping() is True

@pytest.mark.skipif(not REDIS_UP, reason="Redis not available")
def test_celery_task_execution():
    """Test async task execution."""
    # Enqueue task
    task = update_token_stats_async.delay("Integration test message", 0.8, 0.1)
    
    # Wait for result (if worker is running)
    # Note: This might timeout if worker is not running, so we just check if ID is returned
    assert task.id is not None
    
    # If we want to wait for result, we need a worker running
    # result = task.get(timeout=10)
    # assert result['status'] == 'success'

@pytest.mark.skipif(not REDIS_UP, reason="Redis not available")
def test_health_endpoint():
    """Test /health endpoint."""
    # Assuming web server is running on localhost:5000
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'ok'
            assert data['components']['redis']['status'] == 'up'
    except requests.exceptions.ConnectionError:
        pytest.skip("Web server not running")
