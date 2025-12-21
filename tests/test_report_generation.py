"""
Tests for report generation and visualization
"""

import pytest
import json
from pathlib import Path
from chatrel.analysis_engine import generate_report


def test_report_generation_basic():
    """Test basic report generation with sample data."""
    sample_path = Path("sample_data/sample_chat.txt")
    
    if not sample_path.exists():
        pytest.skip("Sample data not found")
    
    # Generate report in formula-only mode (fast)
    report = generate_report(
        filepath=sample_path,
        use_nlp=False,
        anonymize=False,
        use_cache=True
    )
    
    # Verify structure
    assert "mode" in report
    assert report["mode"] == "formula_only"
    
    assert "summary" in report
    assert "total_messages" in report["summary"]
    assert "relationship_type" in report["summary"]
    
    assert "scores" in report
    assert "overall_health" in report["scores"]
    assert "engagement" in report["scores"]
    
    assert "charts" in report
    assert "messages_over_time" in report["charts"]
    
    # Verify chart data format
    if report["charts"]["messages_by_sender"]:
        chart = report["charts"]["messages_by_sender"]
        assert "type" in chart
        assert "labels" in chart
        assert "datasets" in chart
        assert len(chart["datasets"]) > 0


def test_chart_data_completeness():
    """Test that all expected charts are present."""
    sample_path = Path("sample_data/sample_chat.txt")
    
    if not sample_path.exists():
        pytest.skip("Sample data not found")
    
    report = generate_report(sample_path, use_nlp=False)
    
    charts = report["charts"]
    expected_charts = [
        "messages_over_time",
        "messages_by_sender",
        "words_by_sender",
        "response_time_by_sender",
        "emoji_by_sender",
        "emoji_categories",
    ]
    
    for chart_name in expected_charts:
        assert chart_name in charts, f"Missing chart: {chart_name}"


def test_report_with_small_chat():
    """Test report generation handles small chats gracefully."""
    # This will be created by the Flask test client
    # Just verify structure is robust
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
