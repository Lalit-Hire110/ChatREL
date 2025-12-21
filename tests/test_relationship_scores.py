"""
Tests for relationship scoring formulas
"""

import pytest
import numpy as np
from chatrel.relationship_scores import (
    calculate_engagement,
    calculate_warmth,
    calculate_conflict,
    calculate_stability,
    calculate_overall_health,
    predict_relationship_type,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_features_couple():
    """Mock features for a balanced couple relationship."""
    return {
        "global": {
            "total_messages": 500,
            "days_active": 30,
            "unique_senders": 2,
        },
        "per_sender": {
            "Alice": {
                "message_count": 250,
                "word_count": 5000,
                "avg_words_per_message": 20.0,
                "initiation_count": 15,
                "median_response_time_seconds": 300.0,
                "emoji_stats": {
                    "total": 45,
                    "romantic": 12,
                    "positive": 20,
                    "playful": 8,
                    "neutral": 3,
                    "negative": 2,
                },
            },
            "Bob": {
                "message_count": 250,
                "word_count": 4500,
                "avg_words_per_message": 18.0,
                "initiation_count": 18,
                "median_response_time_seconds": 450.0,
                "emoji_stats": {
                    "total": 38,
                    "romantic": 8,
                    "positive": 15,
                    "playful": 10,
                    "neutral": 4,
                    "negative": 1,
                },
            },
        },
    }


@pytest.fixture
def mock_features_one_sided():
    """Mock features for a one-sided relationship."""
    return {
        "global": {
            "total_messages": 100,
            "days_active": 20,
            "unique_senders": 2,
        },
        "per_sender": {
            "Alice": {
                "message_count": 80,
                "word_count": 1600,
                "avg_words_per_message": 20.0,
                "initiation_count": 25,
                "median_response_time_seconds": 200.0,
                "emoji_stats": {
                    "total": 10,
                    "romantic": 3,
                    "positive": 5,
                    "playful": 1,
                    "neutral": 1,
                    "negative": 0,
                },
            },
            "Bob": {
                "message_count": 20,
                "word_count": 300,
                "avg_words_per_message": 15.0,
                "initiation_count": 2,
                "median_response_time_seconds": 3600.0,
                "emoji_stats": {
                    "total": 2,
                    "romantic": 0,
                    "positive": 1,
                    "playful": 1,
                    "neutral": 0,
                    "negative": 0,
                },
            },
        },
    }


@pytest.fixture
def mock_nlp_positive():
    """Mock NLP metrics with positive sentiment."""
    return {
        "sentiment_mean": 0.6,
        "sentiment_std": 0.2,
        "toxicity_mean": 0.1,
        "toxicity_std": 0.05,
        "conflict_flag_count": 2,
        "romantic_keyword_density": 0.08,
        "emoji_affinity": 0.75,
    }


@pytest.fixture
def mock_nlp_negative():
    """Mock NLP metrics with negative sentiment and high conflict."""
    return {
        "sentiment_mean": -0.4,
        "sentiment_std": 0.3,
        "toxicity_mean": 0.7,
        "toxicity_std": 0.2,
        "conflict_flag_count": 25,
        "romantic_keyword_density": 0.01,
        "emoji_affinity": 0.3,
    }


# ============================================================================
# TEST ENGAGEMENT
# ============================================================================

def test_engagement_balanced(mock_features_couple):
    """Test engagement scoring for balanced couple."""
    result = calculate_engagement(mock_features_couple)
    
    assert "score" in result
    assert "normalized" in result
    assert "inputs" in result
    assert "notes" in result
    
    # Check score range
    assert 0 <= result["normalized"] <= 100
    assert result["normalized"] > 50  # Should be moderate-high for balanced couple
    
    # Check inputs
    assert result["inputs"]["total_messages"] == 500
    assert result["inputs"]["days_active"] == 30


def test_engagement_one_sided(mock_features_one_sided):
    """Test engagement scoring for one-sided relationship."""
    result = calculate_engagement(mock_features_one_sided)
    
    assert 0 <= result["normalized"] <= 100
    # Should detect imbalance
    assert any("imbalance" in note.lower() for note in result["notes"])


def test_engagement_small_chat():
    """Test engagement with very few messages."""
    small_features = {
        "global": {"total_messages": 5, "days_active": 2, "unique_senders": 2},
        "per_sender": {
            "A": {"message_count": 3, "initiation_count": 1, "median_response_time_seconds": 100, "avg_words_per_message": 10},
            "B": {"message_count": 2, "initiation_count": 1, "median_response_time_seconds": 150, "avg_words_per_message": 12},
        },
    }
    
    result = calculate_engagement(small_features)
    assert result["normalized"] == 0.0  # Should return 0 for tiny chats
    assert any("unreliable" in note.lower() for note in result["notes"])


# ============================================================================
# TEST WARMTH
# ============================================================================

def test_warmth_formula_only(mock_features_couple):
    """Test warmth scoring without NLP."""
    result = calculate_warmth(mock_features_couple, nlp=None)
    
    assert 0 <= result["normalized"] <= 100
    assert result["used_nlp"] is False
    assert any("formula-only" in note.lower() for note in result["notes"])
    
    # High romantic emoji ratio should result in higher warmth
    assert result["normalized"] > 50


def test_warmth_with_nlp(mock_features_couple, mock_nlp_positive):
    """Test warmth scoring with NLP."""
    result = calculate_warmth(mock_features_couple, nlp=mock_nlp_positive)
    
    assert result["used_nlp"] is True
    assert any("nlp" in note.lower() for note in result["notes"])
    
    # Positive sentiment should boost warmth
    assert result["normalized"] > 50


def test_warmth_toxicity_penalty(mock_features_couple, mock_nlp_negative):
    """Test that high toxicity reduces warmth."""
    result_no_nlp = calculate_warmth(mock_features_couple, nlp=None)
    result_with_toxic = calculate_warmth(mock_features_couple, nlp=mock_nlp_negative)
    
    # Toxicity should reduce warmth
    assert result_with_toxic["normalized"] < result_no_nlp["normalized"]


# ============================================================================
# TEST CONFLICT
# ============================================================================

def test_conflict_low(mock_features_couple):
    """Test conflict scoring for low-conflict relationship."""
    result = calculate_conflict(mock_features_couple, nlp=None)
    
    assert 0 <= result["normalized"] <= 100
    # Low negative emoji ratio should result in low-moderate conflict
    # (mock data has 3/83 = ~3.6% negative emojis)
    assert result["normalized"] < 50  # Should still be in low-moderate range


def test_conflict_with_toxicity(mock_features_couple, mock_nlp_negative):
    """Test conflict scoring with high toxicity."""
    result = calculate_conflict(mock_features_couple, nlp=mock_nlp_negative)
    
    assert result["used_nlp"] is True
    # High toxicity should result in high conflict
    assert result["normalized"] > 50


# ============================================================================
# TEST STABILITY
# ============================================================================

def test_stability_long_term(mock_features_couple):
    """Test stability for longer relationship."""
    result = calculate_stability(mock_features_couple)
    
    assert 0 <= result["normalized"] <= 100
    assert result["normalized"] > 40  # 30 days should be moderate stability


def test_stability_short_term():
    """Test stability for very short relationship."""
    short_features = {
        "global": {"total_messages": 10, "days_active": 1, "unique_senders": 2},
        "per_sender": {},
    }
    
    result = calculate_stability(short_features)
    assert result["normalized"] == 0.0  # Too short to be stable


# ============================================================================
# TEST OVERALL HEALTH
# ============================================================================

def test_overall_health_basic(mock_features_couple):
    """Test overall health calculation."""
    # Calculate sub-scores
    scores = {
        "engagement": calculate_engagement(mock_features_couple),
        "warmth": calculate_warmth(mock_features_couple, nlp=None),
        "conflict": calculate_conflict(mock_features_couple, nlp=None),
        "stability": calculate_stability(mock_features_couple),
    }
    
    result = calculate_overall_health(scores, features=mock_features_couple)
    
    assert "score" in result
    assert "normalized" in result
    assert "confidence" in result
    assert 0 <= result["normalized"] <= 100
    assert 0 <= result["confidence"] <= 1


def test_overall_health_confidence_reduction():
    """Test that confidence reduces for small chats."""
    small_features = {
        "global": {"total_messages": 50, "days_active": 3, "unique_senders": 2},
        "per_sender": {
            "A": {"message_count": 25, "initiation_count": 5, "median_response_time_seconds": 100, "avg_words_per_message": 10, "emoji_stats": {"total": 5, "romantic": 1, "positive": 2, "playful": 1, "neutral": 1, "negative": 0}},
            "B": {"message_count": 25, "initiation_count": 5, "median_response_time_seconds": 100, "avg_words_per_message": 10, "emoji_stats": {"total": 5, "romantic": 1, "positive": 2, "playful": 1, "neutral": 1, "negative": 0}},
        },
    }
    
    scores = {
        "engagement": calculate_engagement(small_features),
        "warmth": calculate_warmth(small_features, nlp=None),
        "conflict": calculate_conflict(small_features, nlp=None),
        "stability": calculate_stability(small_features),
    }
    
    result = calculate_overall_health(scores, features=small_features)
    
    # Confidence should be reduced
    assert result["confidence"] < 1.0
    assert any("message count" in note.lower() or "duration" in note.lower() for note in result["notes"])


# ============================================================================
# TEST RELATIONSHIP TYPE PREDICTION
# ============================================================================

def test_relationship_type_couple(mock_features_couple):
    """Test classification of couple relationship."""
    scores = {
        "engagement": calculate_engagement(mock_features_couple),
        "warmth": calculate_warmth(mock_features_couple, nlp=None),
        "conflict": calculate_conflict(mock_features_couple, nlp=None),
        "stability": calculate_stability(mock_features_couple),
    }
    
    result = predict_relationship_type(scores, mock_features_couple, nlp=None)
    
    assert "type" in result
    assert "confidence" in result
    assert "evidence" in result
    
    # Should classify as couple or romantic
    assert result["type"] in ["couple", "romantic_crush", "friends"]
    assert 0 <= result["confidence"] <= 1


def test_relationship_type_one_sided(mock_features_one_sided):
    """Test classification of one-sided relationship."""
    scores = {
        "engagement": calculate_engagement(mock_features_one_sided),
        "warmth": calculate_warmth(mock_features_one_sided, nlp=None),
        "conflict": calculate_conflict(mock_features_one_sided, nlp=None),
        "stability": calculate_stability(mock_features_one_sided),
    }
    
    result = predict_relationship_type(scores, mock_features_one_sided, nlp=None)
    
    # Should detect as one-sided due to extreme imbalance
    assert result["type"] == "one-sided"
    assert result["confidence"] > 0.6


# ============================================================================
# TEST NORMALIZATION AND CLAMPING
# ============================================================================

def test_score_clamping():
    """Test that all scores are properly clamped to 0-100."""
    extreme_features = {
        "global": {"total_messages": 10000, "days_active": 365, "unique_senders": 2},
        "per_sender": {
            "A": {
                "message_count": 5000,
                "word_count": 100000,
                "avg_words_per_message": 20.0,
                "initiation_count": 500,
                "median_response_time_seconds": 10.0,  # Very fast
                "emoji_stats": {"total": 1000, "romantic": 500, "positive": 400, "playful": 50, "neutral": 40, "negative": 10},
            },
            "B": {
                "message_count": 5000,
                "word_count": 100000,
                "avg_words_per_message": 20.0,
                "initiation_count": 500,
                "median_response_time_seconds": 10.0,
                "emoji_stats": {"total": 1000, "romantic": 500, "positive": 400, "playful": 50, "neutral": 40, "negative": 10},
            },
        },
    }
    
    engagement = calculate_engagement(extreme_features)
    warmth = calculate_warmth(extreme_features, nlp=None)
    conflict = calculate_conflict(extreme_features, nlp=None)
    stability = calculate_stability(extreme_features)
    
    # All scores should be clamped
    assert 0 <= engagement["normalized"] <= 100
    assert 0 <= warmth["normalized"] <= 100
    assert 0 <= conflict["normalized"] <= 100
    assert 0 <= stability["normalized"] <= 100


# ============================================================================
# TEST DETERMINISTIC BEHAVIOR
# ============================================================================

def test_deterministic_scoring(mock_features_couple):
    """Test that scoring is deterministic (same input = same output)."""
    result1 = calculate_engagement(mock_features_couple)
    result2 = calculate_engagement(mock_features_couple)
    
    assert result1["normalized"] == result2["normalized"]
    assert result1["score"] == result2["score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
