"""
Test HuggingFace models directly with real API calls.
Tests sentiment and toxicity models with sample messages.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatrel.hf_client import HFClient
from chatrel import config


def main():
    """Test HF models with sample messages."""
    
    # Test messages (Hinglish + Romanized Marathi)
    messages = [
        "Love you so much 😘",
        "Tu kitna overreact karta hai 😂",
        "Tu khup annoy karte ahes re",
        "You are literally the worst bro",
        "Aaj bara busy ahe, later baat karte hain",
    ]
    
    print("=" * 60)
    print("HuggingFace Model Integration Test")
    print("=" * 60)
    
    # Show configuration
    print("\n--- Configuration ---")
    print(f"Sentiment model: {config.SENTIMENT_MODEL}")
    print(f"Toxicity model: {config.TOXICITY_MODEL}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Using cache: True")
    
    # Validate HF token
    if not config.HF_TOKEN or config.HF_TOKEN == "your_huggingface_token_here":
        print("\n❌ ERROR: HF_TOKEN not set in .env file")
        print("   Get token from: https://huggingface.co/settings/tokens")
        print("   Add to .env: HF_TOKEN=your_token_here")
        return
    
    print(f"HF_TOKEN: {config.HF_TOKEN[:10]}... (valid)")
    
    # Initialize HF client
    print("\n--- Initializing HF Client ---")
    try:
        client = HFClient(mock_mode=False, use_cache=True)
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Test sentiment model
    print("\n--- Testing Sentiment Model ---")
    print(f"Analyzing {len(messages)} messages...")
    
    try:
        sentiments = client.get_sentiment(messages)
        print(f"✓ Got {len(sentiments)} sentiment results")
        
        print("\nSample results:")
        for i, (msg, sent) in enumerate(zip(messages[:3], sentiments[:3]), 1):
            print(f"  {i}. \"{msg[:40]}...\"")
            print(f"     → {sent['label']}: {sent['score']:.3f}")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test toxicity model
    print("\n--- Testing Toxicity Model ---")
    print(f"Analyzing {len(messages)} messages...")
    
    try:
        toxicities = client.get_toxicity(messages)
        print(f"✓ Got {len(toxicities)} toxicity results")
        
        print("\nSample results:")
        for i, (msg, tox) in enumerate(zip(messages[:3], toxicities[:3]), 1):
            print(f"  {i}. \"{msg[:40]}...\"")
            print(f"     → {tox['label']}: {tox['score']:.3f}")
    except Exception as e:
        print(f"❌ Toxicity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Cache statistics
    if client.cache:
        print("\n--- Cache Statistics ---")
        stats = client.cache.stats()
        print(f"Total cached entries: {stats['total_entries']}")
        print(f"By model: {stats['by_model']}")
    
    print("\n" + "=" * 60)
    print("✅ All HF model tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
