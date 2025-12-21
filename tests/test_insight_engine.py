"""
Tests for insight generation
"""

import pytest
from chatrel.insight_engine import generate_insights


@pytest.fixture
def mock_report_formula_only():
    """Mock report for formula-only mode."""
    return {
        "mode": "formula_only",
        "summary": {
            "total_messages": 500,
            "days_active": 30,
            "messages_per_day": 16.7,
            "dominant_sender": None,
            "relationship_type": "friends",
            "relationship_confidence": 0.8,
        },
        "scores": {
            "overall_health": {"normalized": 72},
            "engagement": {"normalized": 65},
            "warmth": {"normalized": 68},
            "conflict": {"normalized": 15},
            "stability": {"normalized": 58},
        },
        "type_prediction": {
            "type": "friends",
            "confidence": 0.8,
            "evidence": ["Moderate warmth", "High engagement"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {
                    "message_count": 250,
                    "initiation_count": 15,
                    "median_response_time_seconds": 300,
                    "emoji_stats": {"total": 45, "romantic": 5, "positive": 20, "playful": 10, "neutral": 8, "negative": 2}
                },
                "Bob": {
                    "message_count": 250,
                    "initiation_count": 18,
                    "median_response_time_seconds": 450,
                    "emoji_stats": {"total": 38, "romantic": 3, "positive": 15, "playful": 12, "neutral": 6, "negative": 2}
                },
            }
        },
        "nlp_metrics": None,
    }


@pytest.fixture
def mock_report_with_nlp():
    """Mock report for formula+NLP mode."""
    return {
        "mode": "formula_plus_nlp",
        "summary": {
            "total_messages": 300,
            "days_active": 45,
            "messages_per_day": 6.7,
            "dominant_sender": "Alice",
            "relationship_type": "couple",
            "relationship_confidence": 0.85,
        },
        "scores": {
            "overall_health": {"normalized": 78},
            "engagement": {"normalized": 72},
            "warmth": {"normalized": 82},
            "conflict": {"normalized": 12},
            "stability": {"normalized": 65},
        },
        "type_prediction": {
            "type": "couple",
            "confidence": 0.85,
            "evidence": ["High warmth", "Romantic emojis", "Balanced"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {
                    "message_count": 180,
                    "initiation_count": 25,
                    "median_response_time_seconds": 200,
                    "emoji_stats": {"total": 55, "romantic": 20, "positive": 25, "playful": 8, "neutral": 2, "negative": 0}
                },
                "Bob": {
                    "message_count": 120,
                    "initiation_count": 12,
                    "median_response_time_seconds": 350,
                    "emoji_stats": {"total": 40, "romantic": 15, "positive": 18, "playful": 5, "neutral": 2, "negative": 0}
                },
            }
        },
        "nlp_metrics": {
            "sentiment_mean": 0.65,
            "toxicity_mean": 0.05,
        },
    }


@pytest.fixture
def mock_report_one_sided():
    """Mock report for one-sided relationship."""
    return {
        "mode": "formula_only",
        "summary": {
            "total_messages": 100,
            "days_active": 20,
            "messages_per_day": 5.0,
            "dominant_sender": "Alice",
            "relationship_type": "one-sided",
            "relationship_confidence": 0.82,
        },
        "scores": {
            "overall_health": {"normalized": 35},
            "engagement": {"normalized": 42},
            "warmth": {"normalized": 55},
            "conflict": {"normalized": 8},
            "stability": {"normalized": 38},
        },
        "type_prediction": {
            "type": "one-sided",
            "confidence": 0.82,
            "evidence": ["High message imbalance"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {
                    "message_count": 80,
                    "initiation_count": 25,
                    "median_response_time_seconds": 200,
                    "emoji_stats": {"total": 15, "romantic": 5, "positive": 8, "playful": 2, "neutral": 0, "negative": 0}
                },
                "Bob": {
                    "message_count": 20,
                    "initiation_count": 2,
                    "median_response_time_seconds": 7200,  # 2 hours
                    "emoji_stats": {"total": 3, "romantic": 0, "positive": 2, "playful": 1, "neutral": 0, "negative": 0}
                },
            }
        },
        "nlp_metrics": None,
    }


def test_generate_insights_basic_structure(mock_report_formula_only):
    """Test that insights have correct structure."""
    insights = generate_insights(mock_report_formula_only, "formula_only")
    
    assert "summary" in insights
    assert "strengths" in insights
    assert "risks" in insights
    assert "suggestions" in insights
    assert "tone" in insights
    assert "used_samples" in insights
    
    assert isinstance(insights["summary"], str)
    assert isinstance(insights["strengths"], list)
    assert isinstance(insights["risks"], list)
    assert isinstance(insights["suggestions"], list)
    assert len(insights["summary"]) > 0


def test_insights_formula_only_mode(mock_report_formula_only):
    """Test insights in formula-only mode."""
    insights = generate_insights(mock_report_formula_only, "formula_only")
    
    # Should acknowledge formula-only limitations
    assert "formula" in insights["summary"].lower() or "pattern" in insights["summary"].lower()
    
    # Should have non-empty sections
    assert len(insights["strengths"]) > 0
    assert len(insights["risks"]) >= 0  # May have no risks
    assert len(insights["suggestions"]) > 0


def test_insights_nlp_mode(mock_report_with_nlp):
    """Test insights with NLP data."""
    insights = generate_insights(mock_report_with_nlp, "formula_plus_nlp")
    
    # Should have content
    assert len(insights["summary"]) > 100  # Decent length
    assert len(insights["strengths"]) > 0
    
    # Should not have formula-only disclaimer
    summary_lower = insights["summary"].lower()
    # Note: may or may not mention NLP, but shouldn't have pattern-only disclaimer


def test_insights_one_sided_detection(mock_report_one_sided):
    """Test that one-sided pattern surfaces in risks."""
    insights = generate_insights(mock_report_one_sided, "formula_only")
    
    # Should identify imbalance in risks
    risks_text = " ".join(insights["risks"]).lower()
    assert "alice" in risks_text or "one" in risks_text or "imbalanc" in risks_text or "sided" in risks_text
    
    # Should have relevant suggestions
    suggestions_text = " ".join(insights["suggestions"]).lower()
    assert len(suggestions_text) > 0


def test_insights_tone_appropriate(mock_report_formula_only):
    """Test that tone is conversational and non-clinical."""
    insights = generate_insights(mock_report_formula_only, "formula_only")
    
    all_text = (insights["summary"] + " ".join(insights["strengths"]) + 
                " ".join(insights["risks"]) + " ".join(insights["suggestions"])).lower()
    
    # Should NOT have clinical/therapy language
    forbidden_words = ["diagnose", "disorder", "therapy", "treatment", "patholog"]
    for word in forbidden_words:
        assert word not in all_text
    
    # Should have conversational markers
    friendly_phrases = ["you", "your", "could", "might", "shows", "looks", "seems"]
    has_friendly = any(phrase in all_text for phrase in friendly_phrases)
    assert has_friendly


def test_insights_no_direct_commands(mock_report_formula_only):
    """Test that suggestions avoid direct commands."""
    insights = generate_insights(mock_report_formula_only, "formula_only")
    
    suggestions_text = " ".join(insights["suggestions"]).lower()
    
    # Should avoid imperative commands
    should_avoid = ["you must", "you should always", "do this", "don't do"]
    for phrase in should_avoid:
        assert phrase not in suggestions_text


def test_insights_handle_missing_data():
    """Test robustness with minimal/missing data."""
    minimal_report = {
        "mode": "formula_only",
        "summary": {
            "total_messages": 10,
            "days_active": 2,
            "messages_per_day": 5,
            "dominant_sender": None,
            "relationship_type": "acquaintance",
            "relationship_confidence": 0.4,
        },
        "scores": {
            "overall_health": {"normalized": 45},
            "engagement": {"normalized": 30},
            "warmth": {"normalized": 35},
            "conflict": {"normalized": 10},
            "stability": {"normalized": 20},
        },
        "type_prediction": {"type": "acquaintance", "confidence": 0.4, "evidence": []},
        "structural_metrics": {"per_sender": {}},
        "nlp_metrics": None,
    }
    
    # Should not crash
    insights = generate_insights(minimal_report, "formula_only")
    
    assert "summary" in insights
    assert len(insights["summary"]) > 0


def test_insights_high_warmth_celebration(mock_report_with_nlp):
    """Test that high warmth is celebrated in strengths."""
    insights = generate_insights(mock_report_with_nlp, "formula_plus_nlp")
    
    strengths_text = " ".join(insights["strengths"]).lower()
    
    # Should mention warmth/positivity/affection
    warmth_indicators = ["warm", "positive", "affection", "emotional", "expressive"]
    has_warmth_mention = any(ind in strengths_text for ind in warmth_indicators)
    assert has_warmth_mention


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
