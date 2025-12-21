"""
End-to-end pipeline integration test.
Tests: Parse → HF API → Processing → Aggregation → Scoring
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatrel.pipeline import run_full_analysis
from chatrel import config


def main():
    """Run full pipeline on sample chat."""
    
    print("=" * 60)
    print("ChatREL v4 - Full Pipeline Integration Test")
    print("=" * 60)
    
    # Locate sample chat
    chat_path = Path("sample_data") / "sample_chat.txt"
    if not chat_path.exists():
        print(f"❌ ERROR: Sample chat not found at {chat_path}")
        print("   Make sure you're running from ChatREL_v4 directory")
        return
    
    print(f"\n✓ Using chat file: {chat_path}")
    
    # Validate HF token
    if not config.HF_TOKEN or config.HF_TOKEN == "your_huggingface_token_here":
        print("\n❌ ERROR: HF_TOKEN not set in .env file")
        print("   Get token from: https://huggingface.co/settings/tokens")
        print("   Add to .env: HF_TOKEN=your_token_here")
        return
    
    try:
        # Run full analysis using shared pipeline
        print("\n--- Running Full Analysis Pipeline ---")
        result = run_full_analysis(
            filepath=chat_path,
            anonymize=False,
            use_cache=True,
        )
        
        report = result['report']
        metrics = result['metrics']
        subscores = result['subscores']
        df_processed = result['df_processed']
        window = result['window']
        
        # Display results
        valid_df = df_processed[~df_processed['is_media'] & df_processed['combined_sentiment'].notna()]
        
        print(f"\n✓ Parsed {len(df_processed)} messages")
        print(f"  Valid text messages: {len(valid_df)}")
        print(f"  Avg sentiment: {valid_df['combined_sentiment'].mean():.3f}")
        print(f"  Avg toxicity: {valid_df['toxicity_score'].mean():.3f}")
        
        print("\nKey metrics:")
        print(f"  Mean sentiment: {metrics['mean_sentiment']:.3f}")
        print(f"  Positive messages: {metrics['percent_positive']:.1f}%")
        print(f"  Negative messages: {metrics['percent_negative']:.1f}%")
        print(f"  Avg toxicity: {metrics['avg_toxicity']:.3f}")
        print(f"  Reciprocity: {metrics['reciprocity']:.3f}")
        print(f"  Emoji affinity: {metrics['emoji_affinity']:.3f}")
        
        print("\nSub-scores (0-1):")
        for name, score in subscores.items():
            print(f"  {name}: {score:.3f}")
        
        print("\n" + "=" * 60)
        print("RELATIONSHIP ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\n🎯 Overall Health: {report['overall_health']}/100")
        print(f"💑 Relationship Type: {report['relationship_type']}")
        print(f"📊 Confidence: {report['relationship_confidence']:.0%}")
        print(f"💭 Reasoning: {report['relationship_reasoning']}")
        
        print("\n📈 Sub-Scores:")
        for name, score in report['sub_scores'].items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            print(f"  {name:12s}: {bar} {score:.1f}/100")
        
        print("\n📊 Scoring Weights:")
        for name, weight in report['scoring_weights'].items():
            print(f"  {name}: {weight:.2f}")
        
        print("\n" + "=" * 60)
        print("✅ Full pipeline test completed successfully!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
