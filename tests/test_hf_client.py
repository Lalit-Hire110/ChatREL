"""
Tests for HF client
"""

import pytest
from chatrel.hf_client import HFClient
from chatrel.cache import ResponseCache


def test_client_init_mock():
    """Test client initialization in mock mode."""
    client = HFClient(mock_mode=True)
    assert client.mock_mode == True


def test_mock_query():
    """Test mock query."""
    client = HFClient(mock_mode=True)
    
    texts = ["I love you", "You're terrible", "Hello world"]
    results = client.get_sentiment(texts)
    
    assert len(results) == len(texts)
    assert all("label" in r for r in results)
    assert all("score" in r for r in results)


def test_batching():
    """Test batching logic."""
    client = HFClient(mock_mode=True, batch_size=2)
    
    texts = ["a", "b", "c", "d", "e"]
    results = client.query(texts, "test_model")
    
    # Should get results for all texts
    assert len(results) == 5


def test_cache_integration():
    """Test cache integration."""
    # Create temporary cache
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(cache_dir=Path(tmpdir), ttl_days=1)
        client = HFClient(mock_mode=True, use_cache=True)
        client.cache = cache
        
        try:
            # First query
            results1 = client.query(["test"], "test_model")
            
            # Second query (should use cache)
            results2 = client.query(["test"], "test_model")
            
            assert results1 == results2
            
            # Check cache has entry
            stats = cache.stats()
            assert stats["total_entries"] > 0
        finally:
            # Explicitly close cache before temp directory cleanup (Windows requirement)
            cache.close()


def test_empty_texts():
    """Test handling empty text list."""
    client = HFClient(mock_mode=True)
    results = client.query([], "test_model")
    assert results == []


def test_sentiment_and_toxicity():
    """Test both sentiment and toxicity models."""
    client = HFClient(mock_mode=True)
    
    texts = ["I love you ❤️", "You're annoying 😡"]
    
    sentiments = client.get_sentiment(texts)
    assert len(sentiments) == 2
    
    toxicities = client.get_toxicity(texts)
    assert len(toxicities) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
