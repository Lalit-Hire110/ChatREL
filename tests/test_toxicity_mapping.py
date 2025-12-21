"""
Unit tests for toxicity score mapping.
Tests that non-toxic predictions are correctly converted to low toxicity scores.
"""

import pytest
from chatrel.message_processor import extract_toxicity_score


class TestToxicityMapping:
    """Test correct mapping of HF toxicity predictions to toxicity scores."""
    
    def test_non_toxic_high_confidence_maps_to_low_toxicity(self):
        """Non-toxic with score 0.97 should give toxicity ≈ 0.03"""
        pred = {"label": "non-toxic", "score": 0.97}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.03, abs=1e-3)
    
    def test_toxic_high_confidence_maps_to_high_toxicity(self):
        """Toxic with score 0.8 should give toxicity = 0.8"""
        pred = {"label": "toxic", "score": 0.8}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.8, abs=1e-3)
    
    def test_non_toxic_low_confidence_maps_to_high_toxicity(self):
        """Non-toxic with low score 0.2 means high uncertainty → toxicity 0.8"""
        pred = {"label": "non-toxic", "score": 0.2}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.8, abs=1e-3)
    
    def test_toxic_low_confidence_maps_to_low_toxicity(self):
        """Toxic with low score 0.1 means low toxicity"""
        pred = {"label": "toxic", "score": 0.1}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.1, abs=1e-3)
    
    def test_hate_label_is_treated_as_toxic(self):
        """Hate label should be treated as toxic"""
        pred = {"label": "hate", "score": 0.7}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.7, abs=1e-3)
    
    def test_offensive_label_is_treated_as_toxic(self):
        """Offensive label should be treated as toxic"""
        pred = {"label": "offensive", "score": 0.6}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.6, abs=1e-3)
    
    def test_clean_label_maps_like_non_toxic(self):
        """Clean label should invert score like non-toxic"""
        pred = {"label": "clean", "score": 0.95}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.05, abs=1e-3)

    def test_neutral_label_maps_like_non_toxic(self):
        """Neutral label should invert score like non-toxic"""
        pred = {"label": "neutral", "score": 0.95}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.05, abs=1e-3)
    
    def test_edge_case_score_zero(self):
        """Non-toxic with score 0 → toxicity 1.0"""
        pred = {"label": "non-toxic", "score": 0.0}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(1.0, abs=1e-3)
    
    def test_edge_case_score_one(self):
        """Non-toxic with score 1.0 → toxicity 0.0"""
        pred = {"label": "non-toxic", "score": 1.0}
        result = extract_toxicity_score(pred)
        assert result == pytest.approx(0.0, abs=1e-3)
    
    def test_missing_label_uses_fallback(self):
        """Unknown label should log warning and use score as-is"""
        pred = {"label": "unknown", "score": 0.5}
        result = extract_toxicity_score(pred)
        # Fallback: uses score as-is
        assert result == pytest.approx(0.5, abs=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
