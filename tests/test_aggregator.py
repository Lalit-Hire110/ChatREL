"""
Tests for aggregator
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from chatrel.aggregator import Aggregator


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    data = {
        "msg_id": range(10),
        "sender": ["Alice", "Bob"] * 5,
        "text": ["Test message"] * 10,
        "timestamp": [datetime.now() - timedelta(hours=i) for i in range(9, -1, -1)],
        "combined_sentiment": [0.5, -0.3, 0.7, -0.1, 0.4, 0.2, -0.5, 0.6, 0.1, 0.3],
        "toxicity_score": [0.1, 0.3, 0.05, 0.2, 0.08, 0.12, 0.6, 0.04, 0.15, 0.09],
        "emoji_count": [1, 0, 2, 1, 3, 0, 1, 2, 1, 1],
        "emoji_valence": [0.5, 0.0, 0.8, 0.3, 0.9, 0.0, -0.4, 0.7, 0.4, 0.5],
        "word_count": [3, 5, 2, 4, 6, 3, 5, 4, 3, 4],
    }
    return pd.DataFrame(data)


def test_message_window(sample_df):
    """Test message window creation."""
    agg = Aggregator()
    
    window = agg.create_message_window(sample_df, window_size=5)
    assert len(window) == 5
    
    # Should take last 5 messages
    assert window["msg_id"].min() == 5


def test_time_window(sample_df):
    """Test time-based window creation."""
    agg = Aggregator()
    
    # All messages within last 24 hours in our sample
    window = agg.create_time_window(sample_df, days=1)
    assert len(window) == len(sample_df)


def test_sentiment_metrics(sample_df):
    """Test sentiment metric computation."""
    agg = Aggregator()
    metrics = agg.compute_metrics(sample_df)
    
    assert "mean_sentiment" in metrics
    assert "median_sentiment" in metrics
    assert "percent_positive" in metrics
    assert "percent_negative" in metrics
    
    # Check ranges
    assert -1 <= metrics["mean_sentiment"] <= 1
    assert 0 <= metrics["percent_positive"] <= 100


def test_toxicity_metrics(sample_df):
    """Test toxicity metric computation."""
    agg = Aggregator()
    metrics = agg.compute_metrics(sample_df)
    
    assert "avg_toxicity" in metrics
    assert "max_toxicity" in metrics
    assert "toxicity_spike_count" in metrics
    
    assert 0 <= metrics["avg_toxicity"] <= 1
    assert metrics["toxicity_spike_count"] >= 0


def test_reciprocity(sample_df):
    """Test reciprocity calculation."""
    agg = Aggregator()
    metrics = agg.compute_metrics(sample_df)
    
    assert "reciprocity" in metrics
    assert 0 <= metrics["reciprocity"] <= 1
    
    # Balanced senders should have high reciprocity
    assert metrics["reciprocity"] > 0.8  # 50/50 split


def test_subscores(sample_df):
    """Test sub-score computation."""
    agg = Aggregator()
    metrics = agg.compute_metrics(sample_df)
    subscores = agg.compute_subscores(metrics)
    
    assert "warmth_score" in subscores
    assert "conflict_score" in subscores
    assert "engagement_score" in subscores
    assert "stability_score" in subscores
    
    # All scores should be 0-1
    for score in subscores.values():
        assert 0 <= score <= 1


def test_empty_df():
    """Test handling empty DataFrame."""
    agg = Aggregator()
    empty_df = pd.DataFrame()
    
    metrics = agg.compute_metrics(empty_df)
    assert isinstance(metrics, dict)
    
    subscores = agg.compute_subscores(metrics)
    assert isinstance(subscores, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
