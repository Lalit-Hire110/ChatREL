"""
Redis client wrapper for ChatREL v4.
Provides robust connection handling, retries, and fallback mechanisms.
"""

import time
import logging
import threading
from typing import Optional
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from chatrel.config import CSM_REDIS_URL

logger = logging.getLogger(__name__)

class RedisUnavailable(Exception):
    """Raised when Redis is not available after retries."""
    pass

class RedisClientWrapper:
    _instance = None
    _lock = threading.Lock()
    _redis_client: Optional[redis.Redis] = None
    _is_connected = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.url = CSM_REDIS_URL
        self._connect()

    def _connect(self):
        """Attempt to connect to Redis with retries."""
        try:
            self._redis_client = redis.Redis.from_url(
                self.url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                decode_responses=True
            )
            # Test connection
            self._redis_client.ping()
            self._is_connected = True
            logger.info(f"Successfully connected to Redis at {self.url}")
        except (ConnectionError, TimeoutError, RedisError) as e:
            self._is_connected = False
            logger.warning(f"Failed to connect to Redis: {e}")
            self._redis_client = None

    def get_client(self) -> Optional[redis.Redis]:
        """
        Get the Redis client.
        
        Returns:
            redis.Redis client if connected, None otherwise.
        """
        if not self._is_connected:
            # Try one quick reconnect attempt if not connected
            self._connect()
        
        return self._redis_client

    def is_available(self) -> bool:
        """Check if Redis is currently available."""
        if not self._is_connected:
            return False
        try:
            self._redis_client.ping()
            return True
        except (ConnectionError, TimeoutError, RedisError):
            self._is_connected = False
            return False

def get_redis_client(max_retries: int = 3) -> Optional[redis.Redis]:
    """
    Get a Redis client with exponential backoff on initial connect.
    
    Args:
        max_retries: Number of retries for initial connection.
        
    Returns:
        redis.Redis client or None if unavailable.
    """
    wrapper = RedisClientWrapper.get_instance()
    client = wrapper.get_client()
    
    if client:
        return client
        
    # If not connected, try with backoff
    for attempt in range(max_retries):
        wait_time = 0.5 * (2 ** attempt)
        time.sleep(wait_time)
        client = wrapper.get_client()
        if client:
            return client
            
    logger.warning("Redis unavailable after retries. Falling back to DB-only mode.")
    return None

def watch_redis():
    """
    Background task to watch Redis connection and reconnect if lost.
    Should be run in a separate thread.
    """
    wrapper = RedisClientWrapper.get_instance()
    while True:
        if not wrapper.is_available():
            logger.info("Redis watcher: attempting to reconnect...")
            wrapper._connect()
        time.sleep(30)
