"""
Unified analysis engine for ChatREL v4
Orchestrates the complete analysis pipeline with mode-aware execution
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .parser import WhatsAppParser
from .hf_client import HFClient
from .message_processor import MessageProcessor
from .aggregator import Aggregator
from .privacy import pseudonymize_dataframe
from .chatstats import extract_structural_features
from .relationship_scores import (
    calculate_engagement,
    calculate_warmth,
    calculate_conflict,
    calculate_stability,
    calculate_overall_health,
    predict_relationship_type,
)
from .utils import demo_cache
from . import config

logger = logging.getLogger(__name__)


def run_analysis(
    filepath: Path,
    use_nlp: Optional[bool] = None,
    anonymize: bool = False,
    use_cache: bool = True,
    mode: str = "normal",
) -> Dict[str, Any]:
    """
    Run analysis pipeline with configurable operational mode.
    
    This is the main entry point for ChatREL analysis. It supports two modes:
    
    1. Formula-Only Mode (use_nlp=False):
       - Fast execution (<2 seconds)
       - No external API dependency
       - Structural metrics only
       - Placeholder scores
       
    2. Formula + NLP Mode (use_nlp=True):
       - Full analysis with sentiment/toxicity
       - Requires HuggingFace API token
       - Combines structural + NLP metrics
       - Complete scoring (after Prompt 2)
    
    Args:
        filepath: Path to WhatsApp export file
        use_nlp: Override config.USE_NLP (None = use config default)
        anonymize: Pseudonymize PII before HF API
        use_cache: Use response cache for HF API
        
    Returns:
        {
            "mode": "formula_only" | "formula_plus_nlp",
            "structural_metrics": Dict from chatstats.extract_structural_features(),
            "nlp_metrics": Dict | None,  # Null when use_nlp=False
            "scores": {
                "engagement": {...},
                "warmth": {...},
                "conflict": {...},
                "stability": {...},
                "overall_health": float | None,  # 0-100 scale
            },
            "metadata": {
                "total_messages": int,
                "senders": List[str],
                "date_range": {...},
                "anonymized": bool,
                "sender_mapping": Dict | None,
            },
        }
        
    Raises:
        ValueError: If use_nlp=True but HF_TOKEN not configured
        FileNotFoundError: If filepath doesn't exist
        Exception: If parsing or analysis fails
    """
    # Determine operational mode
    mode_nlp = use_nlp if use_nlp is not None else config.USE_NLP
    
    # Handle Demo Mode Cache Lookup
    chat_hash = None
    if mode == 'demo' and config.DEMO_CACHE_ENABLED:
        try:
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
            chat_hash = demo_cache.compute_chat_hash(file_bytes)
            
            cached_result = demo_cache.get_demo_result(chat_hash)
            if cached_result:
                logger.info(f"Returning cached demo result for hash {chat_hash[:8]}")
                return cached_result
        except Exception as e:
            logger.warning(f"Demo cache lookup failed: {e}")

    # Handle CSM Mode Override
    use_csm = config.CSM_ENABLED
    if mode == 'csm':
        use_csm = True
        logger.info("Forcing CSM_ENABLED=True for this session (CSM Mode)")
    
    mode_str = "formula_plus_nlp" if mode_nlp else "formula_only"
    
    logger.info(f"Starting analysis in {mode_str} mode")
    logger.info(f"Input file: {filepath}")
    
    # Validate mode-specific requirements
    if mode_nlp and not config.HF_TOKEN:
        raise ValueError(
            "NLP mode requires HF_TOKEN to be configured in .env file. "
            "Get token from: https://huggingface.co/settings/tokens\n"
            "Or set USE_NLP=False for formula-only mode."
        )
    
    # Step 1: Parse chat file
    logger.info("Step 1/5: Parsing chat file")
    parser = WhatsAppParser()
    
    try:
        df = parser.parse_file(str(filepath))
    except Exception as e:
        logger.error(f"Failed to parse chat file: {e}")
        raise
    
    logger.info(f"Parsed {len(df)} messages from {df['sender'].nunique()} senders")
    
    if len(df) == 0:
        raise ValueError("Parsed chat is empty - check file format")
    
    # Step 2: Extract structural features
    logger.info("Step 2/5: Extracting structural features")
    structural_metrics = extract_structural_features(df)
    
    logger.info(
        f"Structural analysis complete: "
        f"{structural_metrics['global']['total_messages']} messages over "
        f"{structural_metrics['global']['days_active']} days"
    )
    
    # Step 3: NLP processing (conditional)
    nlp_metrics = None
    df_processed = df.copy()
    sender_map = {}
    
    if mode_nlp:
        logger.info("Step 3/5: Running NLP analysis (sentiment + toxicity)")
        
        # Pseudonymize if requested
        if anonymize:
            logger.info("Pseudonymizing text before HF API calls")
            df_display, pseudo = pseudonymize_dataframe(df, columns_to_mask=['text'])
            sender_map = pseudo.get_sender_mapping()
        else:
            df_display = df.copy()
        
        # Initialize HF client
        client = HFClient(mock_mode=False, use_cache=use_cache)
        
        # Process messages with CSM if enabled
        if use_csm:
            try:
                from .csm_processor import CSMMessageProcessor
                # Suppress learning if in demo mode and pollution not allowed
                suppress_learning = (mode == 'demo' and not config.DEMO_POLLUTE_CSM)
                
                processor = CSMMessageProcessor(client, use_csm=True, suppress_learning=suppress_learning)
                df_processed = processor.process_messages(df_display)
                logger.info(f"CSM processing enabled (suppress_learning={suppress_learning})")
            except Exception as e:
                logger.warning(f"CSM processing failed: {e} - falling back to standard")
                processor = MessageProcessor(client)
                df_processed = processor.process_messages(df_display)
        else:
            # Standard processing without CSM
            processor = MessageProcessor(client)
            df_processed = processor.process_messages(df_display)

        
        # Aggregate NLP metrics
        aggregator = Aggregator()
        window = aggregator.create_message_window(df_processed)
        raw_metrics = aggregator.compute_metrics(window)
        
        nlp_metrics = {
            "sentiment_mean": raw_metrics.get("sentiment_mean", 0.0),
            "sentiment_std": raw_metrics.get("sentiment_std", 0.0),
            "toxicity_mean": raw_metrics.get("toxicity_mean", 0.0),
            "toxicity_std": raw_metrics.get("toxicity_std", 0.0),
            "conflict_flag_count": raw_metrics.get("conflict_msg_count", 0),
            "romantic_keyword_density": raw_metrics.get("romantic_density", 0.0),
            "emoji_affinity": raw_metrics.get("emoji_affinity", 0.0),
        }
        
        logger.info(
            f"NLP processing complete: "
            f"avg_sentiment={nlp_metrics['sentiment_mean']:.3f}, "
            f"avg_toxicity={nlp_metrics['toxicity_mean']:.3f}"
        )
    else:
        logger.info("Step 3/5: Skipping NLP analysis (formula-only mode)")
    
    # Step 4: Calculate relationship scores
    logger.info("Step 4/5: Computing relationship scores")
    
    scores = {
        "engagement": calculate_engagement(structural_metrics),
        "warmth": calculate_warmth(structural_metrics, nlp=nlp_metrics),
        "conflict": calculate_conflict(structural_metrics, nlp=nlp_metrics),
        "stability": calculate_stability(structural_metrics),
    }
    
    # Compute overall health with confidence
    overall_health_result = calculate_overall_health(
        scores,
        features=structural_metrics,
        nlp=nlp_metrics
    )
    scores["overall_health"] = overall_health_result
    
    # Predict relationship type
    relationship_type = predict_relationship_type(scores, structural_metrics, nlp=nlp_metrics    )
    
    # Log major metrics
    logger.info(f"Scoring complete:")
    logger.info(f"  Engagement: {scores['engagement']['normalized']:.1f}/100")
    logger.info(f"  Warmth: {scores['warmth']['normalized']:.1f}/100")
    logger.info(f"  Conflict: {scores['conflict']['normalized']:.1f}/100")
    logger.info(f"  Stability: {scores['stability']['normalized']:.1f}/100")
    logger.info(f"  Overall Health: {overall_health_result['normalized']:.1f}/100")
    logger.info(f"  Confidence: {overall_health_result['confidence']:.2f}")
    logger.info(f"  Relationship Type: {relationship_type['type']} ({relationship_type['confidence']:.2f})")
    
    # Step 5: Assemble final response
    logger.info("Step 5/5: Assembling response")
    
    result = {
        "mode": mode_str,
        "structural_metrics": structural_metrics,
        "nlp_metrics": nlp_metrics,
        "scores": scores,
        "relationship_type": relationship_type,
        "metadata": {
            "total_messages": len(df),
            "senders": list(sender_map.values()) if anonymize else df["sender"].unique().tolist(),
            "date_range": {
                "start": structural_metrics["global"]["date_range_start"],
                "end": structural_metrics["global"]["date_range_end"],
            },
            "anonymized": anonymize,
            "sender_mapping": sender_map if anonymize else None,
        },
    }
    
    logger.info(f"Analysis complete in {mode_str} mode")
    
    # Cache result if in Demo mode
    if mode == 'demo' and config.DEMO_CACHE_ENABLED and chat_hash:
        try:
            demo_cache.upsert_demo_result(chat_hash, result, mode=mode)
        except Exception as e:
            logger.warning(f"Failed to cache demo result: {e}")
            
    return result


def validate_analysis_result(result: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate the structure of an analysis result.
    
    Args:
        result: Output from run_analysis()
        
    Returns:
        (is_valid, error_message)
    """
    required_keys = ["mode", "structural_metrics", "nlp_metrics", "scores", "metadata"]
    
    for key in required_keys:
        if key not in result:
            return False, f"Missing required key: {key}"
    
    # Validate mode
    if result["mode"] not in ["formula_only", "formula_plus_nlp"]:
        return False, f"Invalid mode: {result['mode']}"
    
    # Validate NLP metrics consistency
    if result["mode"] == "formula_only" and result["nlp_metrics"] is not None:
        return False, "formula_only mode should have nlp_metrics=None"
    
    if result["mode"] == "formula_plus_nlp" and result["nlp_metrics"] is None:
        return False, "formula_plus_nlp mode requires nlp_metrics"
    
    # Validate structural metrics
    if "global" not in result["structural_metrics"]:
        return False, "structural_metrics missing 'global' key"
    
    if "per_sender" not in result["structural_metrics"]:
        return False, "structural_metrics missing 'per_sender' key"
    
    # Validate scores
    required_score_keys = ["engagement", "warmth", "conflict", "stability", "overall_health"]
    for key in required_score_keys:
        if key not in result["scores"]:
            return False, f"scores missing required key: {key}"
    
    return True, "Valid"


def generate_report(
    filepath: Path,
    use_nlp: Optional[bool] = None,
    anonymize: bool = False,
    use_cache: bool = True,
    mode: str = "normal",
) -> Dict[str, Any]:
    """
    Generate complete visualization-ready report for web UI.
    
    This function wraps run_analysis() and adds chart data for frontend consumption.
    
    Args:
        filepath: Path to WhatsApp export file
        use_nlp: Override config.USE_NLP  
        anonymize: Pseudonymize PII
        use_cache: Use HF API cache
        
    Returns:
        Standardized report dict with summary, scores, type_prediction, and charts
    """
    from .report_generator import generate_report_payload
    
    # Run core analysis
    analysis_result = run_analysis(filepath, use_nlp, anonymize, use_cache, mode=mode)
    
    # Re-parse for chart generation (need full dataframe)
    parser = WhatsAppParser()
    df = parser.parse_file(str(filepath))
   
    # Handle anonymization if needed
    sender_map = None
    if anonymize:
        from .privacy import pseudonymize_dataframe
        _, pseudo = pseudonymize_dataframe(df, columns_to_mask=['text'])
        sender_map = pseudo.get_sender_mapping()
    
    # Generate comprehensive report with charts
    report = generate_report_payload(
        df=df,
        structural_metrics=analysis_result["structural_metrics"],
        nlp_metrics=analysis_result["nlp_metrics"],
        scores=analysis_result["scores"],
        relationship_type=analysis_result["relationship_type"],
        mode=analysis_result["mode"],
        sender_mapping=sender_map
    )
    
    return report


if __name__ == "__main__":
    # Test the analysis engine
    import sys
    from pathlib import Path
    
    sample_path = Path("sample_data/sample_chat.txt")
    
    if not sample_path.exists():
        print(f"Sample data not found at {sample_path}")
        sys.exit(1)
    
    print("="*60)
    print("ChatREL v4 Analysis Engine Test")
    print("="*60)
    
    # Test 1: Formula-only mode
    print("\n[Test 1] Formula-Only Mode")
    print("-"*60)
    result_formula = run_analysis(sample_path, use_nlp=False)
    
    print(f"Mode: {result_formula['mode']}")
    print(f"Total messages: {result_formula['metadata']['total_messages']}")
    print(f"Senders: {result_formula['metadata']['senders']}")
    print(f"NLP metrics: {result_formula['nlp_metrics']}")
    print(f"Overall health: {result_formula['scores']['overall_health']}")
    
    is_valid, msg = validate_analysis_result(result_formula)
    print(f"Validation: {msg}")
    
    # Test 2: Formula + NLP mode (if HF_TOKEN is set)
    """
    if config.HF_TOKEN:
        print("\n[Test 2] Formula + NLP Mode")
        print("-"*60)
        result_nlp = run_analysis(sample_path, use_nlp=True)
        
        print(f"Mode: {result_nlp['mode']}")
        print(f"NLP sentiment: {result_nlp['nlp_metrics']['sentiment_mean']:.3f}")
        print(f"NLP toxicity: {result_nlp['nlp_metrics']['toxicity_mean']:.3f}")
        
        is_valid, msg = validate_analysis_result(result_nlp)
        print(f"Validation: {msg}")
    else:
        print("\n[Test 2] Skipped (HF_TOKEN not set)")
    """
    
    print("\n" + "="*60)
    print("✓ Analysis engine functional")
    print("="*60)
