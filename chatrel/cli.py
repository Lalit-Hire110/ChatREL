"""
CLI interface for ChatREL v4
"""

import sys
import json
import logging
import argparse
from pathlib import Path

from . import config
from .parser import WhatsAppParser
from .hf_client import HFClient
from .message_processor import MessageProcessor
from .aggregator import Aggregator
from .scoring import RelationshipScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_file(filepath: str, output_file: str = None, mock: bool = False) -> dict:
    """
    Analyze WhatsApp export file.
    
    Args:
        filepath: Path to WhatsApp .txt export
        output_file: Optional output JSON file
        mock: Use mock HF responses (for testing)
    
    Returns:
        Analysis report dict
    """
    logger.info(f"Analyzing file: {filepath}")
    
    # Validate config
    valid, msg = config.validate_config()
    if not valid and not mock:
        logger.error(f"Configuration error: {msg}")
        sys.exit(1)
    
    # Parse chat
    parser = WhatsAppParser()
    try:
        df = parser.parse_file(filepath)
    except Exception as e:
        logger.error(f"Failed to parse file: {e}")
        sys.exit(1)
    
    logger.info(f"Parsed {len(df)} messages")
    
    # Process messages
    client = HFClient(mock_mode=mock)
    processor = MessageProcessor(client)
    
    try:
        df = processor.process_messages(df)
    except Exception as e:
        logger.error(f"Failed to process messages: {e}")
        sys.exit(1)
    
    # Aggregate metrics
    agg = Aggregator()
    window = agg.create_message_window(df)
    
    metrics = agg.compute_metrics(window)
    subscores = agg.compute_subscores(metrics)
    
    # Score relationship
    scorer = RelationshipScorer()
    report = scorer.generate_report(subscores, metrics)
    
    # Add metadata
    report["metadata"] = {
        "total_messages": len(df),
        "window_size": len(window),
        "senders": df["sender"].unique().tolist(),
        "date_range": {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
        },
    }
    
    # Output
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    else:
        print(json.dumps(report, indent=2))
    
    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ChatREL v4 - WhatsApp Chat Relationship Analyzer"
    )
    
    parser.add_argument(
        "command",
        choices=["analyze", "validate"],
        help="Command to run"
    )
    
    parser.add_argument(
        "filepath",
        help="Path to WhatsApp export .txt file"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock HF API responses (for testing)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == "validate":
        # Just validate format
        from .parser import validate_format
        valid, msg = validate_format(args.filepath)
        print(f"Valid: {valid} - {msg}")
        sys.exit(0 if valid else 1)
    
    elif args.command == "analyze":
        try:
            report = analyze_file(args.filepath, args.output_file, args.mock)
            logger.info("Analysis complete")
            logger.info(f"Overall Health: {report['overall_health']}/100")
            logger.info(f"Relationship Type: {report['relationship_type']}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
