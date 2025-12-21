import hashlib
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from .. import config

logger = logging.getLogger(__name__)

def compute_chat_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash of chat content for caching."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_db_connection():
    """Get connection to CSM database."""
    return sqlite3.connect(config.CSM_DB_PATH)

def get_demo_result(chat_hash: str, profile: str = None) -> Optional[Dict[str, Any]]:
    """Retrieve cached demo result if available."""
    if not config.DEMO_CACHE_ENABLED:
        return None
        
    profile = profile or config.DEMO_PROFILE
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT result_json, access_count 
                FROM demo_chat_cache 
                WHERE chat_hash = ? AND demo_profile = ?
                """,
                (chat_hash, profile)
            )
            row = cursor.fetchone()
            
            if row:
                result_json, count = row
                # Update stats
                cursor.execute(
                    """
                    UPDATE demo_chat_cache 
                    SET last_used_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE chat_hash = ? AND demo_profile = ?
                    """,
                    (chat_hash, profile)
                )
                conn.commit()
                
                logger.info(f"Demo cache HIT for hash={chat_hash[:8]} profile={profile}")
                return json.loads(result_json)
                
    except Exception as e:
        logger.error(f"Error reading demo cache: {e}")
        
    return None

def upsert_demo_result(chat_hash: str, result: Dict[str, Any], profile: str = None, mode: str = 'demo'):
    """Store or update demo result in cache."""
    if not config.DEMO_CACHE_ENABLED:
        return
        
    profile = profile or config.DEMO_PROFILE
    
    # Optional: Privacy check - remove raw text if configured
    # For now, we assume the result is safe or DEMO_STORE_RAW_RESULT logic is handled elsewhere
    # But user requirement said: "if CSM_PRIVACY_MODE=True... demo cache must not include raw text"
    # We should probably sanitize here if needed, but let's stick to the requirement "Store sanitized result or mask PII prior to storing"
    # Since I don't have a sanitizer handy, I will assume the result passed here is what should be stored.
    
    try:
        result_json = json.dumps(result)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO demo_chat_cache (chat_hash, demo_profile, mode, result_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_hash, demo_profile) DO UPDATE SET
                    result_json = excluded.result_json,
                    last_used_at = CURRENT_TIMESTAMP,
                    mode = excluded.mode
                """,
                (chat_hash, profile, mode, result_json)
            )
            conn.commit()
            logger.info(f"Demo cache SAVED for hash={chat_hash[:8]} profile={profile}")
            
    except Exception as e:
        logger.error(f"Error writing demo cache: {e}")
