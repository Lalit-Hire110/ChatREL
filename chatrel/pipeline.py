"""
Shared analysis pipeline for ChatREL v4
Used by both CLI scripts and Flask web interface to ensure consistency
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

from .parser import WhatsAppParser
from .hf_client import HFClient
from .message_processor import MessageProcessor
from .aggregator import Aggregator
from .scoring import RelationshipScorer
from .privacy import pseudonymize_dataframe
from . import config

logger = logging.getLogger(__name__)


def run_full_analysis(
    filepath: Path,
    anonymize: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run complete analysis pipeline on a WhatsApp chat file.
    
    This function encapsulates the entire analysis workflow to ensure
    consistency between CLI and web interface.
    
    Args:
        filepath: Path to WhatsApp export file
        anonymize: Whether to pseudonymize PII before sending to HF API
        use_cache: Whether to use response cache
    
    Returns:
        Dictionary containing:
        - report: Complete relationship analysis report
        - df_processed: Processed DataFrame with annotations
        - sender_map: Mapping of original→pseudonymized names (if anonymize=True)
        - metrics: Raw aggregated metrics
        - subscores: Normalized sub-scores (0-1)
    """
    
    # Step 1: Parse chat
    logger.info(f"Parsing chat file: {filepath}")
    parser = WhatsAppParser()
    df = parser.parse_file(str(filepath))
    logger.info(f"Parsed {len(df)} messages from {df['sender'].nunique()} senders")
    
    # Step 2: Pseudonymize if requested
    sender_map = {}
    if anonymize:
        logger.info("Pseudonymizing chat before HF API calls")
        df_display, pseudo = pseudonymize_dataframe(df, columns_to_mask=['text'])
        sender_map = pseudo.get_sender_mapping()
    else:
        df_display = df.copy()
    
    # Step 3: Initialize HF client (NEVER use mock mode for real analysis)
    if not config.HF_TOKEN or config.HF_TOKEN == "your_huggingface_token_here":
        raise ValueError(
            "HF_TOKEN not configured. Please set it in .env file. "
            "Get token from: https://huggingface.co/settings/tokens"
        )
    
    logger.info("Initializing HuggingFace client (real API mode)")
    client = HFClient(mock_mode=False, use_cache=use_cache)
    
    # Step 4: Process messages with HF models
    logger.info(f"Processing {len(df_display)} messages with HF API")
    processor = MessageProcessor(client)
    df_processed = processor.process_messages(df_display)
    
    # Log some stats
    valid_df = df_processed[~df_processed['is_media'] & df_processed['combined_sentiment'].notna()]
    logger.info(f"Valid messages: {len(valid_df)}")
    logger.info(f"Avg sentiment: {valid_df['combined_sentiment'].mean():.3f}")
    logger.info(f"Avg toxicity: {valid_df['toxicity_score'].mean():.3f}")
    
    # Step 5: Aggregate metrics
    logger.info("Computing aggregate metrics")
    aggregator = Aggregator()
    window = aggregator.create_message_window(df_processed)
    metrics = aggregator.compute_metrics(window)
    subscores = aggregator.compute_subscores(metrics)
    
    # Step 6: Score relationship
    logger.info("Computing final relationship score")
    scorer = RelationshipScorer()
    report = scorer.generate_report(subscores, metrics)
    
    # Add metadata
    report["metadata"] = {
        "total_messages": len(df),
        "window_size": len(window),
        "senders": list(sender_map.values()) if anonymize else df["sender"].unique().tolist(),
        "date_range": {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
        },
        "anonymized": anonymize,
        "sender_mapping": sender_map if anonymize else None,
    }
    
    # Log final scores
    logger.info(f"Analysis complete:")
    logger.info(f"  Overall Health: {report['overall_health']}/100")
    logger.info(f"  Relationship Type: {report['relationship_type']} ({report['relationship_confidence']:.0%})")
    logger.info(f"  Warmth: {report['sub_scores']['warmth']:.1f}/100")
    logger.info(f"  Engagement: {report['sub_scores']['engagement']:.1f}/100")
    logger.info(f"  Conflict: {report['sub_scores']['conflict']:.1f}/100")
    logger.info(f"  Stability: {report['sub_scores']['stability']:.1f}/100")
    
    return {
        "report": report,
        "df_processed": df_processed,
        "sender_map": sender_map,
        "metrics": metrics,
        "subscores": subscores,
        "window": window,
    }


if __name__ == "__main__":
    # Test the pipeline
    from pathlib import Path
    
    chat_path = Path("sample_data/sample_chat.txt")
    
    print("Testing shared pipeline...")
    result = run_full_analysis(chat_path, anonymize=False)
    
    print(f"\nHealth: {result['report']['overall_health']}/100")
    print(f"Type: {result['report']['relationship_type']}")
