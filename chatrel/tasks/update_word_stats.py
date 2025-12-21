"""
Background Tasks for ChatREL v4 - Contextual Sentiment Memory
Handles async token statistics updates and Redis-DB synchronization
"""

import logging
import json
import sqlite3
from pathlib import Path
from typing import List
from datetime import datetime

from ..utils.token_stats import TokenStatsEngine, ContextSignatureGenerator
from .. import config

logger = logging.getLogger(__name__)

# Try to import Celery (optional dependency)
try:
    from celery import Task
    from celery.schedules import crontab
    from .celery_app import celery as app
    
    # Configure periodic tasks
    app.conf.beat_schedule = {
        'sync-redis-to-db': {
            'task': 'chatrel.tasks.update_word_stats.sync_redis_to_db',
            'schedule': crontab(
                minute=0,
                hour=f'*/{config.CSM_REDIS_SYNC_INTERVAL_HOURS}'
            ),
        },
        'cleanup-old-decision-logs': {
            'task': 'chatrel.tasks.update_word_stats.cleanup_decision_logs',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
    }
    
    CELERY_AVAILABLE = True
    logger.info("Celery initialized for async CSM tasks")
    
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning(
        "Celery not available - CSM will run synchronously (performance impact). "
        "Install with: pip install celery"
    )
    app = None


def update_token_stats_sync(text: str, sentiment_score: float, toxicity_score: float):
    """
    Synchronous fallback for token statistics update.
    Called when Celery is not available.
    """
    try:
        engine = TokenStatsEngine()
        tokens = ContextSignatureGenerator.tokenize(text)
        
        for i, token in enumerate(tokens):
            # Update global token stats
            engine.update_token_stats(token, sentiment_score, toxicity_score)
            
            # Update context-specific stats
            signature = ContextSignatureGenerator.generate(tokens, i)
            engine.update_context_stats(token, signature, sentiment_score, toxicity_score)
        
        logger.debug(f"Token stats updated synchronously for {len(tokens)} tokens")
    except Exception as e:
        logger.error(f"Sync token stats update failed: {e}")


# Celery tasks (only defined if Celery is available)
if CELERY_AVAILABLE:
    
    @app.task(bind=True, name='chatrel.tasks.update_word_stats.update_token_stats_async')
    def update_token_stats_async(self, text: str, sentiment_score: float, toxicity_score: float):
        """
        Async task to update token and context statistics.
        
        Args:
            text: Message text
            sentiment_score: Sentiment score from analysis
            toxicity_score: Toxicity score from analysis
        """
        try:
            engine = TokenStatsEngine()
            tokens = ContextSignatureGenerator.tokenize(text)
            
            logger.debug(f"Processing {len(tokens)} tokens asynchronously")
            
            for i, token in enumerate(tokens):
                # Update global token stats
                engine.update_token_stats(token, sentiment_score, toxicity_score)
                
                # Update context-specific stats
                signature = ContextSignatureGenerator.generate(tokens, i)
                engine.update_context_stats(token, signature, sentiment_score, toxicity_score)
            
            logger.debug(f"Async token stats update complete: {len(tokens)} tokens")
            return {
                'status': 'success',
                'tokens_processed': len(tokens),
                'text_preview': text[:50]
            }
            
        except Exception as e:
            from redis.exceptions import RedisError
            if isinstance(e, RedisError):
                logger.warning(f"Redis error in async task: {e} - retrying")
            else:
                logger.error(f"Async token stats update failed: {e}")
            self.retry(exc=e, countdown=60, max_retries=3)
    
    
    @app.task(name='chatrel.tasks.update_word_stats.sync_redis_to_db')
    def sync_redis_to_db():
        """
        Periodic task to sync Redis cache to database.
        Protects against Redis data loss.
        """
        try:
            import redis
            
            redis_client = redis.from_url(config.CSM_REDIS_URL, decode_responses=True)
            db_path = Path(config.CSM_DB_PATH)
            
            tokens_synced = 0
            contexts_synced = 0
            
            # Sync token stats
            for key in redis_client.scan_iter(match="csm:token:*", count=100):
                try:
                    token = key.replace("csm:token:", "")
                    data = json.loads(redis_client.get(key))
                    
                    # Verify DB has this token (no overwrite of more complete data)
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.execute(
                            "SELECT sentiment_count FROM word_stats WHERE token = ?",
                            (token,)
                        )
                        row = cursor.fetchone()
                        
                        if row and row[0] >= data['sentiment_count']:
                            continue  # DB has more recent data
                    
                    tokens_synced += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to sync token {key}: {e}")
            
            # Sync context stats
            for key in redis_client.scan_iter(match="csm:ctx:*", count=100):
                try:
                    parts = key.replace("csm:ctx:", "").split(":", 1)
                    if len(parts) == 2:
                        token, signature = parts
                        data = json.loads(redis_client.get(key))
                        
                        with sqlite3.connect(db_path) as conn:
                            cursor = conn.execute(
                                """
                                SELECT sentiment_count FROM token_context_stats 
                                WHERE token = ? AND context_signature = ?
                                """,
                                (token, signature)
                            )
                            row = cursor.fetchone()
                            
                            if row and row[0] >= data['sentiment_count']:
                                continue
                        
                        contexts_synced += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to sync context {key}: {e}")
            
            # Update sync status
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    UPDATE sync_status
                    SET last_sync_at = CURRENT_TIMESTAMP,
                        tokens_synced = ?,
                        contexts_synced = ?,
                        errors_count = 0
                    WHERE id = 1
                    """,
                    (tokens_synced, contexts_synced)
                )
                conn.commit()
            
            logger.info(
                f"Redis-DB sync complete: {tokens_synced} tokens, "
                f"{contexts_synced} contexts"
            )
            
            return {
                'status': 'success',
                'tokens_synced': tokens_synced,
                'contexts_synced': contexts_synced
            }
            
        except Exception as e:
            logger.error(f"Redis-DB sync failed: {e}")
            
            # Log error in sync_status
            try:
                with sqlite3.connect(Path(config.CSM_DB_PATH)) as conn:
                    conn.execute(
                        """
                        UPDATE sync_status
                        SET errors_count = errors_count + 1,
                            last_error = ?
                        WHERE id = 1
                        """,
                        (str(e),)
                    )
                    conn.commit()
            except Exception:
                pass
            
            raise
    
    
    @app.task(name='chatrel.tasks.update_word_stats.cleanup_decision_logs')
    def cleanup_decision_logs(days: int = 30):
        """
        Periodic task to clean old decision logs.
        
        Args:
            days: Delete logs older than this many days
        """
        try:
            db_path = Path(config.CSM_DB_PATH)
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM decision_log
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                    """,
                    (days,)
                )
                deleted = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleaned {deleted} decision log entries older than {days} days")
            
            return {
                'status': 'success',
                'deleted_entries': deleted
            }
            
        except Exception as e:
            logger.error(f"Decision log cleanup failed: {e}")
            raise


else:
    # Stubs when Celery not available
    update_token_stats_async = None
    sync_redis_to_db = None
    cleanup_decision_logs = None


# Public API
def enqueue_token_update(text: str, sentiment_score: float, toxicity_score: float):
    """
    Enqueue a token statistics update (async if Celery available, sync otherwise).
    
    Args:
        text: Message text
        sentiment_score: Sentiment score
        toxicity_score: Toxicity score
    """
    if not config.CSM_ENABLED:
        return
    
    if CELERY_AVAILABLE and config.CSM_ASYNC_LEARNING and update_token_stats_async:
        try:
            update_token_stats_async.delay(text, sentiment_score, toxicity_score)
            logger.debug("Token update enqueued (async)")
        except Exception as e:
            logger.warning(f"Failed to enqueue async task: {e} - falling back to sync")
            update_token_stats_sync(text, sentiment_score, toxicity_score)
    else:
        if not CELERY_AVAILABLE:
            logger.warning(
                "CSM async learning disabled - Celery not available. "
                "Performance will be degraded. Install Celery for production use."
            )
        update_token_stats_sync(text, sentiment_score, toxicity_score)


def trigger_sync_now():
    """Manually trigger Redis-DB sync (for testing or maintenance)."""
    if CELERY_AVAILABLE and sync_redis_to_db:
        return sync_redis_to_db.delay()
    else:
        logger.warning("Celery not available - cannot trigger async sync")
        return None


if __name__ == "__main__":
    # Test synchronous mode
    print("Testing token stats update...")
    update_token_stats_sync("I love this!", 0.95, 0.05)
    print("Done")
