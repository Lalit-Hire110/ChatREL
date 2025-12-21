"""
Aggregator for ChatREL v4
Windowing and metric computation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from . import config

logger = logging.getLogger(__name__)


class Aggregator:
    """Aggregate message features into relationship metrics."""
    
    def __init__(self):
        pass
    
    def create_message_window(
        self, 
        df: pd.DataFrame, 
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create window of last N messages.
        
        Args:
            df: Full message DataFrame
            window_size: Number of messages (default from config)
        
        Returns:
            Windowed DataFrame
        """
        n = window_size or config.DEFAULT_WINDOW_SIZE
        if len(df) <= n:
            return df.copy()
        return df.tail(n).copy()
    
    def create_time_window(
        self, 
        df: pd.DataFrame, 
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create window of messages from last N days.
        
        Args:
            df: Full message DataFrame
            days: Number of days (default from config)
        
        Returns:
            Windowed DataFrame
        """
        n_days = days or config.DEFAULT_TIME_WINDOW_DAYS
        cutoff = datetime.now() - timedelta(days=n_days)
        
        # Filter by timestamp
        mask = df["timestamp"] >= cutoff
        return df[mask].copy()
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute aggregate metrics for a window.
        
        Returns dict with:
            - sentiment_* metrics
            - toxicity_* metrics
            - emoji_* metrics
            - reciprocity
            - reply time metrics
            - engagement metrics
        """
        if len(df) == 0:
            return self._empty_metrics()
        
        metrics = {}
        
        # Sentiment metrics
        metrics.update(self._compute_sentiment_metrics(df))
        
        # Toxicity metrics
        metrics.update(self._compute_toxicity_metrics(df))
        
        # Emoji metrics
        metrics.update(self._compute_emoji_metrics(df))
        
        # Reciprocity
        metrics.update(self._compute_reciprocity(df))
        
        # Reply times
        metrics.update(self._compute_reply_times(df))
        
        # Engagement
        metrics.update(self._compute_engagement(df))
        
        return metrics
    
    def _compute_sentiment_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute sentiment statistics."""
        valid = df[df["combined_sentiment"].notna()]
        
        if len(valid) == 0:
            return {
                "mean_sentiment": 0.0,
                "median_sentiment": 0.0,
                "sd_sentiment": 0.0,
                "percent_positive": 0.0,
                "percent_negative": 0.0,
            }
        
        sentiments = valid["combined_sentiment"]
        
        return {
            "mean_sentiment": float(sentiments.mean()),
            "median_sentiment": float(sentiments.median()),
            "sd_sentiment": float(sentiments.std()),
            "percent_positive": float((sentiments > 0.2).sum() / len(sentiments) * 100),
            "percent_negative": float((sentiments < -0.2).sum() / len(sentiments) * 100),
        }
    
    def _compute_toxicity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute toxicity statistics."""
        valid = df[df["toxicity_score"].notna()]
        
        if len(valid) == 0:
            return {
                "avg_toxicity": 0.0,
                "max_toxicity": 0.0,
                "toxicity_spike_count": 0,
            }
        
        tox = valid["toxicity_score"]
        
        return {
            "avg_toxicity": float(tox.mean()),
            "max_toxicity": float(tox.max()),
            "toxicity_spike_count": int((tox > 0.7).sum()),
        }
    
    def _compute_emoji_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute emoji statistics."""
        valid = df[df["emoji_valence"].notna()]
        
        if len(valid) == 0:
            return {
                "emoji_density": 0.0,
                "emoji_affinity": 0.5,
            }
        
        total_emojis = df["emoji_count"].sum()
        total_msgs = len(df)
        
        # Affinity = ratio of positive emojis
        positive_emojis = valid[valid["emoji_valence"] > 0.3]["emoji_count"].sum()
        total_counted_emojis = valid["emoji_count"].sum()
        
        affinity = positive_emojis / total_counted_emojis if total_counted_emojis > 0 else 0.5
        
        return {
            "emoji_density": float(total_emojis / total_msgs) if total_msgs > 0 else 0.0,
            "emoji_affinity": float(affinity),
        }
    
    def _compute_reciprocity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute reciprocity metrics."""
        sender_counts = df["sender"].value_counts()
        
        if len(sender_counts) < 2:
            return {"reciprocity": 1.0}
        
        # Reciprocity = 1 - imbalance
        top_two = sender_counts.iloc[:2]
        total = top_two.sum()
        imbalance = abs(top_two.iloc[0] - top_two.iloc[1]) / total if total > 0 else 0
        
        return {
            "reciprocity": float(1.0 - imbalance),
        }
    
    def _compute_reply_times(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute reply time statistics."""
        if len(df) < 2:
            return {
                "median_reply_time_sec": 0.0,
                "percent_replies_under_1h": 0.0,
            }
        
        # Compute time differences between consecutive messages
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["time_diff"] = df_sorted["timestamp"].diff().dt.total_seconds()
        
        # Only consider differences between different senders (actual replies)
        df_sorted["sender_changed"] = df_sorted["sender"] != df_sorted["sender"].shift(1)
        replies = df_sorted[df_sorted["sender_changed"] & df_sorted["time_diff"].notna()]
        
        if len(replies) == 0:
            return {
                "median_reply_time_sec": 0.0,
                "percent_replies_under_1h": 0.0,
            }
        
        reply_times = replies["time_diff"]
        
        # Filter outliers (> 24 hours treated as new conversation)
        reply_times = reply_times[reply_times < 86400]
        
        if len(reply_times) == 0:
            return {
                "median_reply_time_sec": 0.0,
                "percent_replies_under_1h": 0.0,
            }
        
        return {
            "median_reply_time_sec": float(reply_times.median()),
            "percent_replies_under_1h": float((reply_times < 3600).sum() / len(reply_times) * 100),
        }
    
    def _compute_engagement(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute engagement metrics."""
        if len(df) == 0:
            return {
                "msgs_per_day": 0.0,
                "avg_words": 0.0,
            }
        
        # Messages per day
        date_range = (df["timestamp"].max() - df["timestamp"].min()).days + 1
        msgs_per_day = len(df) / date_range if date_range > 0 else 0
        
        # Average words
        avg_words = df["word_count"].mean()
        
        return {
            "msgs_per_day": float(msgs_per_day),
            "avg_words": float(avg_words),
        }
    
    def compute_subscores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Convert raw metrics into normalized sub-scores (0-1).
        
        Returns:
            - warmth_score
            - conflict_score
            - engagement_score
            - stability_score
        """
        # Warmth (positive sentiment + emoji affinity)
        warmth = (
            0.5 * self._normalize(metrics["percent_positive"], 0, 80, clip=True)
            + 0.3 * metrics["emoji_affinity"]
            + 0.2 * self._normalize(metrics["mean_sentiment"], -0.5, 0.5, clip=True)
        )
        
        # Conflict (toxicity + negative sentiment)
        conflict = (
            0.5 * metrics["avg_toxicity"]
            + 0.3 * self._normalize(metrics["percent_negative"], 0, 50, clip=True)
            + 0.2 * min(1.0, metrics["toxicity_spike_count"] / 5.0)
        )
        
        # Engagement (messages per day + reply speed)
        engagement = (
            0.5 * self._normalize(metrics["msgs_per_day"], 0, 20, clip=True)
            + 0.3 * metrics["reciprocity"]
            + 0.2 * self._normalize(metrics["percent_replies_under_1h"], 0, 80, clip=True)
        )
        
        # Stability (low variance in sentiment)
        stability = 1.0 - min(1.0, metrics["sd_sentiment"] / 0.5)
        
        return {
            "warmth_score": float(max(0, min(1, warmth))),
            "conflict_score": float(max(0, min(1, conflict))),
            "engagement_score": float(max(0, min(1, engagement))),
            "stability_score": float(max(0, min(1, stability))),
        }
    
    def _normalize(self, value: float, min_val: float, max_val: float, clip: bool = True) -> float:
        """Normalize value from [min_val, max_val] to [0, 1]."""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        
        if clip:
            return max(0.0, min(1.0, normalized))
        return normalized
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            "mean_sentiment": 0.0,
            "median_sentiment": 0.0,
            "sd_sentiment": 0.0,
            "percent_positive": 0.0,
            "percent_negative": 0.0,
            "avg_toxicity": 0.0,
            "max_toxicity": 0.0,
            "toxicity_spike_count": 0,
            "emoji_density": 0.0,
            "emoji_affinity": 0.5,
            "reciprocity": 1.0,
            "median_reply_time_sec": 0.0,
            "percent_replies_under_1h": 0.0,
            "msgs_per_day": 0.0,
            "avg_words": 0.0,
        }


if __name__ == "__main__":
    # Test aggregator
    from .parser import WhatsAppParser
    from .hf_client import HFClient
    from .message_processor import MessageProcessor
    
    # Load sample data
    parser = WhatsAppParser()
    sample_path = config.PROJECT_ROOT / "sample_data" / "sample_chat.txt"
    
    if sample_path.exists():
        df = parser.parse_file(str(sample_path))
        
        client = HFClient(mock_mode=True)
        processor = MessageProcessor(client)
        df = processor.process_messages(df)
        
        agg = Aggregator()
        window = agg.create_message_window(df, window_size=50)
        
        metrics = agg.compute_metrics(window)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
        
        subscores = agg.compute_subscores(metrics)
        print("\nSub-scores:")
        for k, v in subscores.items():
            print(f"  {k}: {v:.2f}")
    else:
        print(f"Sample data not found at {sample_path}")
