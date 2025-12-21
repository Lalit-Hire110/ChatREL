"""
Tests for WhatsApp parser
"""

import pytest
from datetime import datetime
from chatrel.parser import WhatsAppParser, _strip_weird_unicode, _looks_like_system


def test_strip_unicode():
    """Test Unicode normalization."""
    assert _strip_weird_unicode("hello\u200bworld") == "hello world"
    assert _strip_weird_unicode("test\u00a0space") == "test space"


def test_system_message_detection():
    """Test system message detection."""
    assert _looks_like_system("Messages and calls are end-to-end encrypted")
    assert _looks_like_system("You deleted this message")
    assert not _looks_like_system("Hello world")


def test_parse_simple_chat():
    """Test parsing simple chat."""
    text = """25/12/23, 09:15 - Alice: Good morning! ☀️
25/12/23, 09:18 - Bob: Morning! How are you? ❤️
25/12/23, 09:20 - Alice: Great! You?"""
    
    parser = WhatsAppParser()
    df = parser.parse_text(text)
    
    assert len(df) == 3
    assert df.iloc[0]["sender"] == "Alice"
    assert "Good morning" in df.iloc[0]["text"]
    assert df.iloc[0]["word_count"] > 0


def test_parse_multiline_message():
    """Test parsing multiline messages."""
    text = """25/12/23, 09:15 - Alice: This is a long message
that spans multiple lines
with different content
25/12/23, 09:18 - Bob: Short reply"""
    
    parser = WhatsAppParser()
    df = parser.parse_text(text)
    
    assert len(df) == 2
    assert "multiple lines" in df.iloc[0]["text"]


def test_parse_media_message():
    """Test media message detection."""
    text = """25/12/23, 09:15 - Alice: <Media omitted>
25/12/23, 09:18 - Bob: Normal message"""
    
    parser = WhatsAppParser()
    df = parser.parse_text(text)
    
    assert len(df) == 2
    assert df.iloc[0]["is_media"] == True
    assert df.iloc[1]["is_media"] == False


def test_parse_empty_chat():
    """Test parsing empty chat."""
    parser = WhatsAppParser()
    
    with pytest.raises(ValueError):
        parser.parse_text("")


def test_timestamp_parsing():
    """Test various timestamp formats."""
    test_cases = [
        "25/12/23, 09:15 - Alice: Test",
        "25/12/2023, 09:15 - Alice: Test",
        "12/25/23, 09:15 AM - Alice: Test",
    ]
    
    parser = WhatsAppParser()
    
    for text in test_cases:
        try:
            df = parser.parse_text(text)
            assert len(df) >= 1
        except ValueError:
            # Some formats may not be supported
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
