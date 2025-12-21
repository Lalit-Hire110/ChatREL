"""
Cache module for ChatREL v4
SQLite-based cache for HuggingFace API responses
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from . import config

logger = logging.getLogger(__name__)


class ResponseCache:
    """SQLite-based cache for API responses with TTL."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_days: Optional[int] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache database (default from config)
            ttl_days: TTL in days (default from config)
        """
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_days = ttl_days or config.CACHE_TTL_DAYS
        self.db_path = self.cache_dir / "responses.db"
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
            conn.commit()
        logger.debug(f"Cache database initialized at {self.db_path}")
    
    def _make_key(self, model_name: str, text: str) -> str:
        """Generate cache key from model name and text."""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    
    def get(self, model_name: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for model/text pair.
        
        Returns None if not found or expired.
        """
        key = self._make_key(model_name, text)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, timestamp FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
        
        if row is None:
            logger.debug(f"Cache miss for {model_name}:{text[:50]}...")
            return None
        
        response_json, timestamp_str = row
        timestamp = datetime.fromisoformat(timestamp_str)
        
        # Check TTL
        if datetime.now() - timestamp > timedelta(days=self.ttl_days):
            logger.debug(f"Cache expired for {model_name}:{text[:50]}...")
            self.delete(model_name, text)
            return None
        
        logger.debug(f"Cache hit for {model_name}:{text[:50]}...")
        return json.loads(response_json)
    
    def set(self, model_name: str, text: str, response: Dict[str, Any]):
        """Store response in cache."""
        key = self._make_key(model_name, text)
        timestamp = datetime.now().isoformat()
        response_json = json.dumps(response)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, model_name, text, response, timestamp) VALUES (?, ?, ?, ?, ?)",
                (key, model_name, text, response_json, timestamp)
            )
            conn.commit()
        
        logger.debug(f"Cached response for {model_name}:{text[:50]}...")
    
    def delete(self, model_name: str, text: str):
        """Delete specific cache entry."""
        key = self._make_key(model_name, text)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
    
    def clear_expired(self):
        """Remove all expired entries."""
        cutoff = (datetime.now() - timedelta(days=self.ttl_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared {deleted} expired cache entries")
        return deleted
    
    def clear_all(self):
        """Clear entire cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache")
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared all {deleted} cache entries")
        return deleted
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            total = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT model_name, COUNT(*) FROM cache GROUP BY model_name")
            by_model = dict(cursor.fetchall())
            
            cutoff = (datetime.now() - timedelta(days=self.ttl_days)).isoformat()
            cursor = conn.execute("SELECT COUNT(*) FROM cache WHERE timestamp < ?", (cutoff,))
            expired = cursor.fetchone()[0]
        
        return {
            "total_entries": total,
            "by_model": by_model,
            "expired": expired,
            "ttl_days": self.ttl_days,
            "db_path": str(self.db_path),
        }
    
    def close(self):
        """
        Close any open connections and clean up resources.
        
        This is important on Windows where open file handles prevent deletion.
        """
        import gc
        
        # Force garbage collection to close any lingering connection objects
        gc.collect()
        
        # Force close any lingering connections by connecting and closing
        # This ensures all SQLite connections are properly released
        try:
            if self.db_path.exists():
                # Create a connection and immediately close it to flush any pending operations
                conn = sqlite3.connect(self.db_path)
                conn.close()
                
                # Force another GC pass after explicit close
                gc.collect()
        except Exception as e:
            logger.debug(f"Error during cache close: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass


if __name__ == "__main__":
    # Test cache
    cache = ResponseCache()
    print("Cache stats:", json.dumps(cache.stats(), indent=2))
    
    # Test set/get
    cache.set("test_model", "hello world", {"label": "positive", "score": 0.95})
    result = cache.get("test_model", "hello world")
    print("Retrieved:", result)
    
    # Test expiry
    print("Cleared expired:", cache.clear_expired())
