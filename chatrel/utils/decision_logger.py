"""
Decision Logger for ChatREL v4 - Contextual Sentiment Memory
Logs decision traces for debugging and performance analysis
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .. import config

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Logs CSM decision traces when CSM_DEBUG_DECISIONS=True.
    Useful for debugging confidence issues and performance tuning.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(config.CSM_DB_PATH)
        self.enabled = config.CSM_DEBUG_DECISIONS
        
        if not self.enabled:
            logger.debug("Decision logging disabled (CSM_DEBUG_DECISIONS=False)")
    
    def log_decision(
        self,
        message_hash: str,
        resolution_source: str,
        confidence_score: float,
        variance_factor: float = 1.0,
        context_matches: int = 0,
        token_matches: int = 0,
        unknown_tokens: int = 0,
        decision_reason: str = "",
        lookup_time_ms: float = 0.0,
        inference_time_ms: float = 0.0,
        hf_api_time_ms: float = 0.0,
        total_time_ms: float = 0.0
    ):
        """
        Log a decision trace entry.
        
        Args:
            message_hash: Hash of the message
            resolution_source: 'cache' | 'inference' | 'hybrid' | 'hf'
            confidence_score: Final confidence score
            variance_factor: Variance adjustment factor
            context_matches: Number of context signature matches
            token_matches: Number of global token matches
            unknown_tokens: Number of unknown tokens
            decision_reason: Human-readable reason for decision
            lookup_time_ms: Time spent on cache lookup
            inference_time_ms: Time spent on inference
            hf_api_time_ms: Time spent on HF API call
            total_time_ms: Total time for this message
        """
        if not self.enabled:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO decision_log
                    (message_hash, resolution_source, confidence_score, variance_factor,
                     context_matches_count, token_matches_count, unknown_tokens_count,
                     decision_reason, lookup_time_ms, inference_time_ms, hf_api_time_ms,
                     total_time_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        message_hash, resolution_source, confidence_score, variance_factor,
                        context_matches, token_matches, unknown_tokens,
                        decision_reason, lookup_time_ms, inference_time_ms, hf_api_time_ms,
                        total_time_ms
                    )
                )
                conn.commit()
            
            logger.debug(
                f"Decision logged: {resolution_source} (conf={confidence_score:.2f}, "
                f"var_factor={variance_factor:.2f}, total_time={total_time_ms:.1f}ms)"
            )
            
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")
    
    def get_summary_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics from decision log.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            Statistics dict
        """
        if not self.enabled:
            return {'error': 'Decision logging disabled'}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total decisions by source
                cursor = conn.execute(
                    """
                    SELECT resolution_source, COUNT(*), AVG(confidence_score), 
                           AVG(total_time_ms)
                    FROM decision_log
                    WHERE created_at > datetime('now', '-' || ? || ' hours')
                    GROUP BY resolution_source
                    """,
                    (hours,)
                )
                by_source = {
                    row[0]: {
                        'count': row[1],
                        'avg_confidence': row[2],
                        'avg_time_ms': row[3]
                    }
                    for row in cursor.fetchall()
                }
                
                # Overall stats
                cursor = conn.execute(
                    """
                    SELECT COUNT(*), AVG(confidence_score), AVG(variance_factor),
                           AVG(total_time_ms), MIN(total_time_ms), MAX(total_time_ms)
                    FROM decision_log
                    WHERE created_at > datetime('now', '-' || ? || ' hours')
                    """,
                    (hours,)
                )
                overall = cursor.fetchone()
                
                return {
                    'time_window_hours': hours,
                    'total_decisions': overall[0] if overall else 0,
                    'avg_confidence': overall[1] if overall else 0,
                    'avg_variance_factor': overall[2] if overall else 0,
                    'avg_latency_ms': overall[3] if overall else 0,
                    'min_latency_ms': overall[4] if overall else 0,
                    'max_latency_ms': overall[5] if overall else 0,
                    'by_source': by_source
                }
                
        except Exception as e:
            logger.error(f"Failed to get decision log stats: {e}")
            return {'error': str(e)}


# Timing context manager for easy timing
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000


if __name__ == "__main__":
    # Test decision logging
    logger_inst = DecisionLogger()
    
    logger_inst.log_decision(
        message_hash="test123",
        resolution_source="inference",
        confidence_score=0.85,
        variance_factor=0.92,
        context_matches=5,
        token_matches=8,
        unknown_tokens=2,
        decision_reason="High confidence inference",
        lookup_time_ms=2.5,
        inference_time_ms=45.3,
        hf_api_time_ms=0.0,
        total_time_ms=47.8
    )
    
    print("Stats:", logger_inst.get_summary_stats(hours=1))
