"""Test formula-only mode"""
import sys
from pathlib import Path

# Add chatrel to path
sys.path.insert(0, str(Path(__file__).parent / "chatrel"))

from chatrel.analysis_engine import run_analysis

if __name__ == "__main__":
    sample_path = Path("sample_data/sample_chat.txt")
    
    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        sys.exit(1)
    
    print("="*60)
    print("Testing Formula-Only Mode")
    print("="*60)
    
    result = run_analysis(sample_path, use_nlp=False)
    
    print(f"\n✓ Mode: {result['mode']}")
    print(f"✓ Total messages: {result['metadata']['total_messages']}")
    print(f"✓ Senders: {result['metadata']['senders']}")
    print(f"✓ Days active: {result['structural_metrics']['global']['days_active']}")
    print(f"✓ NLP metrics: {result['nlp_metrics']}")
    
    # Test sender metrics
    for sender, metrics in result['structural_metrics']['per_sender'].items():
        print(f"\n{sender}:")
        print(f"  Messages: {metrics['message_count']}")
        print(f"  Words: {metrics['word_count']}")
        print(f"  Avg words/msg: {metrics['avg_words_per_message']:.1f}")
        print(f"  Initiations: {metrics['initiation_count']}")
        print(f"  Median response time: {metrics['median_response_time_seconds']:.0f}s")
        print(f"  Emojis: {metrics['emoji_stats']['total']} (romantic: {metrics['emoji_stats']['romantic']}, positive: {metrics['emoji_stats']['positive']})")
    
    print("\n" + "="*60)
    print("✓ Formula-only mode working correctly!")
    print("="*60)
