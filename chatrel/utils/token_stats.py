"""
Token Statistics Module for ChatREL v4 - Contextual Sentiment Memory
Implements Welford's algorithm for incremental statistics and heuristic inference
"""

import json
import logging
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import emoji

from .. import config

logger = logging.getLogger(__name__)

# Try to import Redis client wrapper
try:
    from .redis_client import get_redis_client
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False



class WelfordStats:
    """Welford's algorithm for incremental mean and variance calculation."""
    
    @staticmethod
    def update(
        current_mean: float,
        current_m2: float,
        current_count: int,
        new_value: float
    ) -> Tuple[float, float, int, float]:
        """
        Update statistics with a new value using Welford's algorithm.
        
        Returns:
            (new_mean, new_m2, new_count, new_variance)
        """
        count = current_count + 1
        delta = new_value - current_mean
        mean = current_mean + delta / count
        delta2 = new_value - mean
        m2 = current_m2 + delta * delta2
        variance = m2 / count if count > 1 else 0.0
        
        return mean, m2, count, variance


class ContextSignatureGenerator:
    """Generates context signatures for tokens based on surrounding context."""
    
    # Negation words
    NEGATIONS = {
        'not', 'never', 'no', 'neither', 'nor', 'none', 'nobody', 'nothing',
        'nowhere', 'hardly', 'barely', 'scarcely', "don't", "doesn't", "didn't",
        "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "isn't",
        "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization (whitespace + basic punctuation)."""
        # Extract emojis first
        emojis = emoji.emoji_list(text)
        emoji_chars = {e['emoji'] for e in emojis}
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # Add emojis back
        for e in emoji_chars:
            tokens.append(e)
        
        return tokens
    
    @staticmethod
    def has_negation_nearby(tokens: List[str], index: int, window: int = 3) -> bool:
        """Check if there's a negation word within ±window tokens."""
        start = max(0, index - window)
        end = min(len(tokens), index + window + 1)
        
        for i in range(start, end):
            if i != index and tokens[i] in ContextSignatureGenerator.NEGATIONS:
                return True
        return False
    
    @staticmethod
    def has_emoji_in_context(tokens: List[str]) -> bool:
        """Check if message contains emojis."""
        for token in tokens:
            if token in config.EMOJI_LEXICON:
                return True
        return False
    
    @staticmethod
    def generate(tokens: List[str], index: int) -> str:
        """
        Generate context signature for token at index.
        
        Examples:
            - "NEG_good" (negation before "good")
            - "CAPS_AMAZING" (all caps)
            - "EMO_love" (emoji present in message)
            - "PREV_very_good" (previous token "very")
        """
        if not tokens or index >= len(tokens):
            return ""
        
        token = tokens[index]
        signature_parts = []
        
        # Check negation window
        if ContextSignatureGenerator.has_negation_nearby(tokens, index):
            signature_parts.append("NEG")
        
        # Check capitalization (use original case from full text if available)
        # For now, we don't have original case, so skip this
        
        # Check emoji context
        if ContextSignatureGenerator.has_emoji_in_context(tokens):
            signature_parts.append("EMO")
        
        # Add previous token if available and significant
        if index > 0 and tokens[index - 1].isalnum():
            prev = tokens[index - 1]
            if len(prev) > 2:  # Skip very short words
                signature_parts.append(f"PREV_{prev}")
        
        # Build final signature
        if signature_parts:
            return "_".join(signature_parts + [token])
        else:
            return token  # No special context


class TokenStatsEngine:
    """
    Token-level statistics engine with variance-aware confidence adjustment.
    Supports heuristic inference using token and context statistics.
    """
    
    def __init__(self, db_path: Optional[Path] = None, redis_url: Optional[str] = None):
        self.db_path = db_path or Path(config.CSM_DB_PATH)
        
        # Initialize Redis (optional)
        self.redis_client = None
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Use the robust client wrapper
            self.redis_client = get_redis_client(max_retries=1)
            if self.redis_client:
                logger.info("TokenStatsEngine: Redis connected via wrapper")
            else:
                logger.warning("TokenStatsEngine: Redis unavailable - using DB-only mode")
        
        logger.info(f"TokenStatsEngine initialized (redis={'yes' if self.redis_client else 'no'})")
    
    def update_token_stats(
        self,
        token: str,
        sentiment: float,
        toxicity: float
    ):
        """
        Update global token statistics using Welford's algorithm.
        
        Args:
            token: Normalized token
            sentiment: Sentiment score (-1 to 1 or 0 to 1 depending on model)
            toxicity: Toxicity score (0 to 1)
        """
        token = token.lower().strip()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute(
                """
                SELECT sentiment_mean, sentiment_m2, sentiment_count,
                       toxicity_mean, toxicity_m2, toxicity_count
                FROM word_stats
                WHERE token = ?
                """,
                (token,)
            )
            row = cursor.fetchone()
            
            if row:
                sent_mean, sent_m2, sent_count, tox_mean, tox_m2, tox_count = row
            else:
                sent_mean = sent_m2 = sent_count = 0
                tox_mean = tox_m2 = tox_count = 0
            
            # Update sentiment stats
            new_sent_mean, new_sent_m2, new_sent_count, new_sent_var = WelfordStats.update(
                sent_mean, sent_m2, sent_count, sentiment
            )
            
            # Update toxicity stats
            new_tox_mean, new_tox_m2, new_tox_count, new_tox_var = WelfordStats.update(
                tox_mean, tox_m2, tox_count, toxicity
            )
            
            # Check emoji boost
            emoji_boost = config.EMOJI_LEXICON.get(token, 0.0)
            
            # Check stability (variance threshold)
            is_stable = 1 if new_sent_var < config.CSM_VARIANCE_THRESHOLD else 0
            
            # Insert or update
            conn.execute(
                """
                INSERT INTO word_stats
                (token, sentiment_mean, sentiment_m2, sentiment_count, sentiment_variance,
                 toxicity_mean, toxicity_m2, toxicity_count, toxicity_variance,
                 emoji_boost, is_stable, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(token) DO UPDATE SET
                    sentiment_mean = ?,
                    sentiment_m2 = ?,
                    sentiment_count = ?,
                    sentiment_variance = ?,
                    toxicity_mean = ?,
                    toxicity_m2 = ?,
                    toxicity_count = ?,
                    toxicity_variance = ?,
                    is_stable = ?,
                    last_updated_at = CURRENT_TIMESTAMP
                """,
                (
                    token, new_sent_mean, new_sent_m2, new_sent_count, new_sent_var,
                    new_tox_mean, new_tox_m2, new_tox_count, new_tox_var,
                    emoji_boost, is_stable,
                    # Update values
                    new_sent_mean, new_sent_m2, new_sent_count, new_sent_var,
                    new_tox_mean, new_tox_m2, new_tox_count, new_tox_var,
                    is_stable
                )
            )
            conn.commit()
        
        # Update Redis cache
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Attempt lazy reconnect if needed
            if self.redis_client is None:
                self.redis_client = get_redis_client(max_retries=0)
                
            if self.redis_client:
                try:
                    key = f"csm:token:{token}"
                    data = {
                        'sentiment_mean': new_sent_mean,
                        'sentiment_count': new_sent_count,
                        'sentiment_variance': new_sent_var,
                        'toxicity_mean': new_tox_mean,
                        'is_stable': bool(is_stable)
                    }
                    self.redis_client.setex(
                        key,
                        config.CSM_REDIS_TTL_HOURS * 3600,
                        json.dumps(data)
                    )
                except Exception as e:
                    logger.debug(f"Redis update failed for token {token}: {e}")
    
    def update_context_stats(
        self,
        token: str,
        context_signature: str,
        sentiment: float,
        toxicity: float
    ):
        """Update context-specific statistics."""
        token = token.lower().strip()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute(
                """
                SELECT sentiment_mean, sentiment_m2, sentiment_count,
                       toxicity_mean, toxicity_m2, toxicity_count
                FROM token_context_stats
                WHERE token = ? AND context_signature = ?
                """,
                (token, context_signature)
            )
            row = cursor.fetchone()
            
            if row:
                sent_mean, sent_m2, sent_count, tox_mean, tox_m2, tox_count = row
            else:
                sent_mean = sent_m2 = sent_count = 0
                tox_mean = tox_m2 = tox_count = 0
            
            # Update stats
            new_sent_mean, new_sent_m2, new_sent_count, new_sent_var = WelfordStats.update(
                sent_mean, sent_m2, sent_count, sentiment
            )
            new_tox_mean, new_tox_m2, new_tox_count, new_tox_var = WelfordStats.update(
                tox_mean, tox_m2, tox_count, toxicity
            )
            
            is_stable = 1 if new_sent_var < config.CSM_VARIANCE_THRESHOLD else 0
            
            conn.execute(
                """
                INSERT INTO token_context_stats
                (token, context_signature, 
                 sentiment_mean, sentiment_m2, sentiment_count, sentiment_variance,
                 toxicity_mean, toxicity_m2, toxicity_count, toxicity_variance,
                 is_stable, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(token, context_signature) DO UPDATE SET
                    sentiment_mean = ?,
                    sentiment_m2 = ?,
                    sentiment_count = ?,
                    sentiment_variance = ?,
                    toxicity_mean = ?,
                    toxicity_m2 = ?,
                    toxicity_count = ?,
                    toxicity_variance = ?,
                    is_stable = ?,
                    last_updated_at = CURRENT_TIMESTAMP
                """,
                (
                    token, context_signature,
                    new_sent_mean, new_sent_m2, new_sent_count, new_sent_var,
                    new_tox_mean, new_tox_m2, new_tox_count, new_tox_var,
                    is_stable,
                    # Update values
                    new_sent_mean, new_sent_m2, new_sent_count, new_sent_var,
                    new_tox_mean, new_tox_m2, new_tox_count, new_tox_var,
                    is_stable
                )
            )
            conn.commit()
        
        # Update Redis
        if REDIS_AVAILABLE and config.CSM_ENABLED:
            # Attempt lazy reconnect if needed
            if self.redis_client is None:
                self.redis_client = get_redis_client(max_retries=0)
                
            if self.redis_client:
                try:
                    key = f"csm:ctx:{token}:{context_signature}"
                    data = {
                        'sentiment_mean': new_sent_mean,
                        'sentiment_count': new_sent_count,
                        'sentiment_variance': new_sent_var,
                        'is_stable': bool(is_stable)
                    }
                    self.redis_client.setex(
                        key,
                        config.CSM_REDIS_TTL_HOURS * 3600,
                        json.dumps(data)
                    )
                except Exception as e:
                    logger.debug(f"Redis context update failed: {e}")
    
    def infer_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Infer sentiment for text using token and context statistics.
        
        Returns:
            {
                'score': float,  # Estimated sentiment score
                'confidence': float,  # Confidence (0-1)
                'variance_factor': float,  # Variance penalty
                'context_matches': int,
                'token_matches': int,
                'unknown_tokens': int
            }
        """
        tokens = ContextSignatureGenerator.tokenize(text)
        
        if not tokens:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'variance_factor': 1.0,
                'context_matches': 0,
                'token_matches': 0,
                'unknown_tokens': 0
            }
        
        weighted_scores = []
        variance_sum = 0.0
        variance_count = 0
        context_matches = 0
        token_matches = 0
        unknown_tokens = 0
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            signature = ContextSignatureGenerator.generate(tokens, i)
            
            score = None
            variance = 0.0
            weight = 0.1  # Default for unknown
            
            # Try context-specific lookup first
            if REDIS_AVAILABLE and config.CSM_ENABLED:
                if self.redis_client is None:
                    self.redis_client = get_redis_client(max_retries=0)

                if self.redis_client:
                    try:
                        key = f"csm:ctx:{token_lower}:{signature}"
                        cached = self.redis_client.get(key)
                        if cached:
                            data = json.loads(cached)
                            if data['sentiment_count'] >= config.CSM_MIN_CONTEXT_COUNT:
                                score = data['sentiment_mean']
                                variance = data.get('sentiment_variance', 0.0)
                                weight = math.log(1 + data['sentiment_count']) * 2.0
                                context_matches += 1
                    except Exception:
                        pass
            
            # Fallback to DB context lookup
            if score is None:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT sentiment_mean, sentiment_variance, sentiment_count
                        FROM token_context_stats
                        WHERE token = ? AND context_signature = ?
                        """,
                        (token_lower, signature)
                    )
                    row = cursor.fetchone()
                    
                    if row and row[2] >= config.CSM_MIN_CONTEXT_COUNT:
                        score = row[0]
                        variance = row[1]
                        weight = math.log(1 + row[2]) * 2.0
                        context_matches += 1
            
            # Try global token stats
            if score is None:
                if REDIS_AVAILABLE and config.CSM_ENABLED:
                    if self.redis_client is None:
                        self.redis_client = get_redis_client(max_retries=0)

                    if self.redis_client:
                        try:
                            key = f"csm:token:{token_lower}"
                            cached = self.redis_client.get(key)
                            if cached:
                                data = json.loads(cached)
                                if data['sentiment_count'] >= config.CSM_MIN_TOKEN_COUNT:
                                    score = data['sentiment_mean']
                                    variance = data.get('sentiment_variance', 0.0)
                                    weight = math.log(1 + data['sentiment_count'])
                                    token_matches += 1
                        except Exception:
                            pass
                
                # DB fallback
                if score is None:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            """
                            SELECT sentiment_mean, sentiment_variance, sentiment_count, emoji_boost
                            FROM word_stats
                            WHERE token = ?
                            """,
                            (token_lower,)
                        )
                        row = cursor.fetchone()
                        
                        if row and row[2] >= config.CSM_MIN_TOKEN_COUNT:
                            score = row[0]
                            variance = row[1]
                            weight = math.log(1 + row[2])
                            token_matches += 1
            
            # Check emoji lexicon
            if score is None and token in config.EMOJI_LEXICON:
                score = config.EMOJI_LEXICON[token]
                weight = 1.5
                token_matches += 1
            
            # Unknown token
            if score is None:
                score = 0.0
                weight = 0.1
                unknown_tokens += 1
            
            weighted_scores.append((score, weight))
            if variance > 0:
                variance_sum += variance
                variance_count += 1
        
        # Calculate weighted mean
        total_weight = sum(w for _, w in weighted_scores)
        if total_weight > 0:
            weighted_mean = sum(s * w for s, w in weighted_scores) / total_weight
        else:
            weighted_mean = 0.0
        
        # Calculate confidence based on resolved tokens
        resolved = context_matches + token_matches
        raw_confidence = resolved / len(tokens) if tokens else 0.0
        
        # Calculate variance penalty
        avg_variance = variance_sum / variance_count if variance_count > 0 else 0.0
        normalized_variance = min(avg_variance, 1.0)
        variance_factor = 1.0 - (normalized_variance * config.CSM_VAR_CONFIDENCE_WEIGHT)
        
        # Apply variance penalty to confidence
        effective_confidence = raw_confidence * variance_factor
        
        return {
            'score': weighted_mean,
            'confidence': effective_confidence,
            'variance_factor': variance_factor,
            'context_matches': context_matches,
            'token_matches': token_matches,
            'unknown_tokens': unknown_tokens
        }
    
    def infer_toxicity(self, text: str) -> Dict[str, Any]:
        """Infer toxicity (similar logic to sentiment)."""
        tokens = ContextSignatureGenerator.tokenize(text)
        
        if not tokens:
            return {'score': 0.0, 'confidence': 0.0, 'variance_factor': 1.0}
        
        weighted_scores = []
        variance_sum = 0.0
        variance_count = 0
        resolved = 0
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            signature = ContextSignatureGenerator.generate(tokens, i)
            
            score = None
            variance = 0.0
            weight = 0.1
            
            # Context lookup
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT toxicity_mean, toxicity_variance, toxicity_count
                    FROM token_context_stats
                    WHERE token = ? AND context_signature = ?
                    """,
                    (token_lower, signature)
                )
                row = cursor.fetchone()
                
                if row and row[2] >= config.CSM_MIN_CONTEXT_COUNT:
                    score = row[0]
                    variance = row[1]
                    weight = math.log(1 + row[2]) * 2.0
                    resolved += 1
            
            # Global token lookup
            if score is None:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT toxicity_mean, toxicity_variance, toxicity_count
                        FROM word_stats
                        WHERE token = ?
                        """,
                        (token_lower,)
                    )
                    row = cursor.fetchone()
                    
                    if row and row[2] >= config.CSM_MIN_TOKEN_COUNT:
                        score = row[0]
                        variance = row[1]
                        weight = math.log(1 + row[2])
                        resolved += 1
            
            if score is None:
                score = 0.0  # Default: non-toxic
            
            weighted_scores.append((score, weight))
            if variance > 0:
                variance_sum += variance
                variance_count += 1
        
        total_weight = sum(w for _, w in weighted_scores)
        weighted_mean = sum(s * w for s, w in weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        raw_confidence = resolved / len(tokens) if tokens else 0.0
        avg_variance = variance_sum / variance_count if variance_count > 0 else 0.0
        variance_factor = 1.0 - (min(avg_variance, 1.0) * config.CSM_VAR_CONFIDENCE_WEIGHT)
        effective_confidence = raw_confidence * variance_factor
        
        return {
            'score': weighted_mean,
            'confidence': effective_confidence,
            'variance_factor': variance_factor
        }
    
    def get_token_coverage(self) -> Dict[str, Any]:
        """Get statistics on token coverage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM word_stats")
            total_tokens = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM word_stats WHERE sentiment_count >= ?",
                (config.CSM_MIN_TOKEN_COUNT,)
            )
            usable_tokens = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM token_context_stats")
            total_contexts = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM word_stats WHERE is_stable = 0"
            )
            unstable_tokens = cursor.fetchone()[0]
        
        return {
            'total_tokens': total_tokens,
            'usable_tokens': usable_tokens,
            'total_contexts': total_contexts,
            'unstable_tokens': unstable_tokens,
            'coverage_ratio': usable_tokens / total_tokens if total_tokens > 0 else 0.0
        }


if __name__ == "__main__":
    # Test
    engine = TokenStatsEngine()
    
    # Test inference
    result = engine.infer_sentiment("I love this so much!")
    print("Sentiment inference:", json.dumps(result, indent=2))
    
    # Coverage
    print("Coverage:", json.dumps(engine.get_token_coverage(), indent=2))
