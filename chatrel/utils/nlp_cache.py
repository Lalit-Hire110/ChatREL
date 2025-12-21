"""
NLP Cache Module for ChatREL v4 - Contextual Sentiment Memory
Handles message-level caching with version-aware invalidation and privacy modes
"""

import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from .. import config

logger = logging.getLogger(__name__)

# Try to import Redis client wrapper
try:
    from .redis_client import get_redis_client
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis client wrapper not found - CSM will use DB-only mode")


class RateLimiter:
    """Rolling window rate limiter for HuggingFace API calls."""
    
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = deque()  # Timestamps of recent calls
        self.throttled = False
        
    def acquire(self) -> bool:
        """
        Check if a call is allowed. Returns True if allowed, False if throttled.
        """
        now = time.time()
        
        # Remove calls outside the window
        while self.calls and self.calls[0] < now - self.window_seconds:
            self.calls.popleft()
        
        # Check if we're under the limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            if self.throttled:
                logger.info("Rate limit recovered - resuming HF API calls")
                self.throttled = False
            return True
        else:
            if not self.throttled:
                logger.warning(
                    f"HF API rate limit reached ({self.max_calls}/{self.window_seconds}s) - "
                    "switching to inference-only mode"
                )
                self.throttled = True
            return False
    
    def is_throttled(self) -> bool:
        """Check if currently throttled without acquiring."""
        now = time.time()
        while self.calls and self.calls[0] < now - self.window_seconds:
            self.calls.popleft()
        return len(self.calls) >= self.max_calls


class NLPCache:
    """
    Message-level cache with hash-based lookup, version tracking, and privacy mode.
    Supports Redis (hot cache) + SQLite (persistence) with rate limiting.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
        privacy_mode: Optional[bool] = None,
        enable_rate_limit: bool = True
    ):
        """
        Initialize NLP cache.
        
        Args:
            db_path: Path to SQLite database
            redis_url: Redis connection string (ignored if using redis_client wrapper)
            privacy_mode: Store only hashes (no raw text)
            enable_rate_limit: Enable HF API rate limiting
        """
        self.db_path = db_path or Path(config.CSM_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.privacy_mode = privacy_mode if privacy_mode is not None else config.CSM_PRIVACY_MODE
        
        # Initialize database
        self._init_db()
        
        # Initialize Redis (optional)
        self.redis_client = None
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Use the robust client wrapper
            self.redis_client = get_redis_client(max_retries=1)
            if self.redis_client:
                logger.info("Redis connected via wrapper")
            else:
                logger.warning("Redis unavailable - using DB-only mode")
        
        # Initialize rate limiter
        self.rate_limiter = None
        if enable_rate_limit:
            self.rate_limiter = RateLimiter(
                max_calls=config.MAX_HF_CALLS_PER_MIN,
                window_seconds=config.HF_THROTTLE_WINDOW_SECONDS
            )
        
        logger.info(
            f"NLPCache initialized (privacy={self.privacy_mode}, "
            f"redis={'yes' if self.redis_client else 'no'}, "
            f"rate_limit={'yes' if self.rate_limiter else 'no'})"
        )
    
    def _init_db(self):
        """Initialize database schema if not exists."""
        schema_path = Path(__file__).parent.parent.parent / "database" / "schema_csm.sql"
        
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()
            logger.debug(f"CSM database initialized: {self.db_path}")
        else:
            logger.warning(f"Schema file not found: {schema_path} - using minimal schema")
            # Minimal fallback schema
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS message_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text_hash TEXT UNIQUE NOT NULL,
                        text_sanitized TEXT,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        sentiment_label TEXT,
                        sentiment_score REAL,
                        toxicity_label TEXT,
                        toxicity_score REAL,
                        confidence REAL DEFAULT 1.0,
                        source TEXT DEFAULT 'hf',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                conn.commit()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent hashing.
        
        - Lowercase
        - Strip whitespace
        - Remove duplicate spaces/newlines
        - Normalize unicode
        """
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)  # Max 3 repeats
        return text
    
    def hash_text(self, text: str) -> str:
        """Generate SHA-256 hash of normalized text."""
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def sanitize_text(self, text: str) -> Optional[str]:
        """
        Sanitize text for storage (privacy-aware).
        
        In privacy mode: Returns None
        In normal mode: Returns lowercase text with PII patterns masked
        """
        if self.privacy_mode:
            return None
        
        # Basic PII masking (phone numbers, emails)
        text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
        text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
        
        return text[:500]  # Limit length
    
    def get(
        self,
        text: str,
        model_name: str,
        model_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a message.
        
        Returns:
            Dict with 'sentiment' and 'toxicity' scores, or None if not cached
        """
        text_hash = self.hash_text(text)
        
        # Try Redis first (hot cache)
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Attempt lazy reconnect if needed
            if self.redis_client is None:
                self.redis_client = get_redis_client(max_retries=0)

            if self.redis_client:
                try:
                    key = f"csm:msg:{model_name}:{model_version}:{text_hash}"
                    cached = self.redis_client.get(key)
                    if cached:
                        logger.debug(f"Redis cache hit: {text[:50]}...")
                        return json.loads(cached)
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
                    # Don't set self.redis_client to None here, let the wrapper handle it or just fail this once
        
        # Fallback to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT sentiment_label, sentiment_score, toxicity_label, toxicity_score,
                       confidence, source
                FROM message_cache
                WHERE text_hash = ? AND model_name = ? AND model_version = ?
                """,
                (text_hash, model_name, model_version)
            )
            row = cursor.fetchone()
        
        if row:
            # Update access count
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE message_cache
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE text_hash = ? AND model_name = ? AND model_version = ?
                    """,
                    (text_hash, model_name, model_version)
                )
                conn.commit()
            
            result = {
                'sentiment': {'label': row[0], 'score': row[1]},
                'toxicity': {'label': row[2], 'score': row[3]},
                'confidence': row[4],
                'source': row[5]
            }
            
            # Cache in Redis for future hits
            if self.redis_client:
                try:
                    key = f"csm:msg:{model_name}:{model_version}:{text_hash}"
                    self.redis_client.setex(
                        key,
                        config.CSM_REDIS_TTL_HOURS * 3600,
                        json.dumps(result)
                    )
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
            
            logger.debug(f"DB cache hit: {text[:50]}...")
            return result
        
        logger.debug(f"Cache miss: {text[:50]}...")
        return None
    
    def set(
        self,
        text: str,
        model_name: str,
        model_version: str,
        sentiment: Dict[str, Any],
        toxicity: Dict[str, Any],
        confidence: float = 1.0,
        source: str = 'hf'
    ):
        """
        Store analysis result in cache.
        
        Args:
            text: Original message text
            model_name: Model identifier
            model_version: Model version
            sentiment: Sentiment result dict
            toxicity: Toxicity result dict
            confidence: Confidence score (0-1)
            source: Resolution source ('hf', 'inference', 'hybrid')
        """
        text_hash = self.hash_text(text)
        text_sanitized = self.sanitize_text(text)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO message_cache
                (text_hash, text_sanitized, model_name, model_version,
                 sentiment_label, sentiment_score, toxicity_label, toxicity_score,
                 confidence, source, created_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
                """,
                (
                    text_hash,
                    text_sanitized,
                    model_name,
                    model_version,
                    sentiment.get('label'),
                    sentiment.get('score'),
                    toxicity.get('label'),
                    toxicity.get('score'),
                    confidence,
                    source
                )
            )
            conn.commit()
        
        # Store in Redis
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Attempt lazy reconnect if needed
            if self.redis_client is None:
                self.redis_client = get_redis_client(max_retries=0)
                
            if self.redis_client:
                try:
                    key = f"csm:msg:{model_name}:{model_version}:{text_hash}"
                    result = {
                        'sentiment': sentiment,
                        'toxicity': toxicity,
                        'confidence': confidence,
                        'source': source
                    }
                    self.redis_client.setex(
                        key,
                        config.CSM_REDIS_TTL_HOURS * 3600,
                        json.dumps(result)
                    )
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
        
        logger.debug(f"Cached: {text[:50]}... (source={source}, conf={confidence:.2f})")
    
    def can_call_hf_api(self) -> bool:
        """Check if HF API call is allowed (rate limiting)."""
        if not self.rate_limiter:
            return True
        return self.rate_limiter.acquire()
    
    def is_hf_throttled(self) -> bool:
        """Check if HF API is currently throttled."""
        if not self.rate_limiter:
            return False
        return self.rate_limiter.is_throttled()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM message_cache")
            total = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT source, COUNT(*) FROM message_cache GROUP BY source"
            )
            by_source = dict(cursor.fetchall())
            
            cursor = conn.execute(
                "SELECT AVG(confidence) FROM message_cache"
            )
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            cursor = conn.execute(
                "SELECT SUM(access_count) FROM message_cache"
            )
            total_hits = cursor.fetchone()[0] or 0
        
        stats = {
            'total_entries': total,
            'by_source': by_source,
            'avg_confidence': avg_confidence,
            'total_cache_hits': total_hits,
            'redis_connected': self.redis_client is not None,
            'privacy_mode': self.privacy_mode
        }
        
        if self.rate_limiter:
            stats['hf_throttled'] = self.rate_limiter.is_throttled()
            stats['hf_calls_in_window'] = len(self.rate_limiter.calls)
        
        return stats
    
    def clear_old_entries(self, days: int = 30):
        """Remove cache entries older than N days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM message_cache
                WHERE created_at < datetime('now', '-' || ? || ' days')
                """,
                (days,)
            )
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared {deleted} cache entries older than {days} days")
        return deleted


if __name__ == "__main__":
    # Test the cache
    cache = NLPCache()
    
    test_text = "I love you so much!"
    
    # Test set
    cache.set(
        test_text,
        "sentiment_model",
        "1.0",
        sentiment={'label': 'positive', 'score': 0.95},
        toxicity={'label': 'non-toxic', 'score': 0.05},
        confidence=1.0,
        source='hf'
    )
    
    # Test get
    result = cache.get(test_text, "sentiment_model", "1.0")
    print("Retrieved:", result)
    
    # Test stats
    print("Stats:", json.dumps(cache.stats(), indent=2))
