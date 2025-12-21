"""
Tests for query engine
"""

import pytest
from chatrel.query_engine import classify_intent, answer_query


# ============================================================================
# TEST INTENT CLASSIFICATION
# ============================================================================

def test_classify_intent_explain_score():
    """Test explanation intent detection."""
    tests = [
        ("What is engagement?", "EXPLAIN_SCORE", "engagement"),
        ("Explain warmth score", "EXPLAIN_SCORE", "warmth"),
        ("What does stability mean?", "EXPLAIN_SCORE", "stability"),
    ]
    
    for question, expected_intent, expected_target in tests:
        intent, target = classify_intent(question)
        assert intent == expected_intent
        assert target == expected_target


def test_classify_intent_why_low():
    """Test why-low detection."""
    tests = [
        ("Why is my stability low?", "WHY_LOW_SCORE", "stability"),
        ("Why is engagement bad?", "WHY_LOW_SCORE", "engagement"),
        ("Why is warmth so poor?", "WHY_LOW_SCORE", "warmth"),
    ]
    
    for question, expected_intent, expected_target in tests:
        intent, target = classify_intent(question)
        assert intent == expected_intent
        assert target == expected_target


def test_classify_intent_who_texts_more():
    """Test who-texts-more detection."""
    tests = [
        "Who texts more?",
        "Who messages more?",
        "Am I putting more effort?",
        "Is this one-sided?",
    ]
    
    for question in tests:
        intent, target = classify_intent(question)
        assert intent == "WHO_TEXTS_MORE"


def test_classify_intent_timing():
    """Test timing pattern detection."""
    tests = [
        "Who replies faster?",
        "Do they take long to reply?",
        "Who has faster response times?",
    ]
    
    for question in tests:
        intent, target = classify_intent(question)
        assert intent == "TIMING_PATTERN"


def test_classify_intent_out_of_scope():
    """Test out-of-scope detection."""
    tests = [
        "Should I break up?",
        "Does she love me?",
        "Will this relationship last?",
        "Should I marry them?",
    ]
    
    for question in tests:
        intent, target = classify_intent(question)
        assert intent == "OUT_OF_SCOPE"


def test_classify_intent_help():
    """Test help detection."""
    tests = [
        "What can you tell me?",
        "What can I ask you?",
        "Help",
    ]
    
    for question in tests:
        intent, target = classify_intent(question)
        assert intent == "HELP_CAPABILITIES"


# ============================================================================
# TEST ANSWER GENERATION
# ============================================================================

@pytest.fixture
def mock_report():
    """Mock report for testing."""
    return {
        "summary": {
            "total_messages": 150,
            "days_active": 20,
            "messages_per_day": 7.5,
            "dominant_sender": "Alice",
        },
        "scores": {
            "engagement": {
                "normalized": 55,
                "inputs": {"msgs_per_day": 7.5, "balance_score": 65}
            },
            "warmth": {
                "normalized": 68,
                "inputs": {"emoji_affection_ratio": 0.4, "avg_words_per_message": 15}
            },
            "conflict": {
                "normalized": 15,
                "inputs": {}
            },
            "stability": {
                "normalized": 35,
                "inputs": {}
            },
            "overall_health": {
                "normalized": 58
            },
        },
        "type_prediction": {
            "type": "friends",
            "confidence": 0.75,
            "evidence": ["Moderate warmth", "Balanced messaging"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {
                    "message_count": 95,
                    "median_response_time_seconds": 300
                },
                "Bob": {
                    "message_count": 55,
                    "median_response_time_seconds": 1800
                },
            }
        }
    }


def test_answer_explain_score(mock_report):
    """Test explaining a score."""
    result = answer_query(mock_report, "What is engagement?")
    
    assert result["intent"] == "EXPLAIN_SCORE"
    assert result["target"] == "engagement"
    assert len(result["answer"]) > 0
    assert "engagement" in result["answer"].lower()
    assert "55" in result["answer"]  # Should mention the score
    assert 0 <= result["confidence"] <= 1
    assert len(result["provenance"]) > 0


def test_answer_why_low_score(mock_report):
    """Test explaining why a score is low."""
    result = answer_query(mock_report, "Why is my stability low?")
    
    assert result["intent"] == "WHY_LOW_SCORE"
    assert result["target"] == "stability"
    assert len(result["answer"]) > 0
    assert "low" in result["answer"].lower()
    assert len(result["provenance"]) > 0


def test_answer_who_texts_more(mock_report):
    """Test who texts more."""
    result = answer_query(mock_report, "Who texts more?")
    
    assert result["intent"] == "WHO_TEXTS_MORE"
    assert len(result["answer"]) > 0
    assert "Alice" in result["answer"] or "bob" in result["answer"].lower()
    assert len(result["provenance"]) >= 2  # Should have both senders


def test_answer_timing_pattern(mock_report):
    """Test timing pattern answer."""
    result = answer_query(mock_report, "Who replies faster?")
    
    assert result["intent"] == "TIMING_PATTERN"
    assert len(result["answer"]) > 0
    assert "minutes" in result["answer"].lower() or "hours" in result["answer"].lower()


def test_answer_out_of_scope(mock_report):
    """Test out-of-scope handling."""
    result = answer_query(mock_report, "Should I break up?")
    
    assert result["intent"] == "OUT_OF_SCOPE"
    assert len(result["answer"]) > 0
    assert "can't" in result["answer"].lower() or "cannot" in result["answer"].lower()
    assert result["error"] is not None


def test_answer_help(mock_report):
    """Test help response."""
    result = answer_query(mock_report, "What can you tell me?")
    
    assert result["intent"] == "HELP_CAPABILITIES"
    assert len(result["answer"]) > 0
    assert "ask" in result["answer"].lower()


def test_answer_structure(mock_report):
    """Test that all answers have required structure."""
    questions = [
        "What is warmth?",
        "Why is stability low?",
        "Who texts more?",
    ]
    
    for q in questions:
        result = answer_query(mock_report, q)
        
        # Required keys
        assert "answer" in result
        assert "intent" in result
        assert "target" in result
        assert "provenance" in result
        assert "confidence" in result
        assert "error" in result
        
        # Types
        assert isinstance(result["answer"], str)
        assert isinstance(result["intent"], str)
        assert isinstance(result["provenance"], list)
        assert isinstance(result["confidence"], float)
        
        # Ranges
        assert 0 <= result["confidence"] <= 1
        assert len(result["answer"]) > 0


def test_provenance_format(mock_report):
    """Test provenance has correct format."""
    result = answer_query(mock_report, "What is engagement?")
    
    assert len(result["provenance"]) > 0
    
    for item in result["provenance"]:
        assert "type" in item
        assert "key" in item
        assert "value" in item


def test_no_crash_on_edge_cases():
    """Test robustness with minimal data."""
    minimal_report = {
        "summary": {"total_messages": 10, "days_active": 2},
        "scores": {},
        "type_prediction": {"type": "unknown", "confidence": 0.1, "evidence": []},
        "structural_metrics": {"per_sender": {}}
    }
    
    questions = [
        "What is engagement?",
        "Who texts more?",
        "Help",
    ]
    
    for q in questions:
        result = answer_query(minimal_report, q)
        assert "answer" in result
        assert len(result["answer"]) > 0  # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
