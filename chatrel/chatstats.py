"""
Structural chat statistics module for ChatREL v4
Extracts behavioral features without NLP dependency
"""

import logging
from typing import Dict, Any
from datetime import timedelta
import pandas as pd
import numpy as np
import emoji

from . import config

logger = logging.getLogger(__name__)


def extract_structural_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract structural behavioral features from parsed messages.
    
    This function performs fast, deterministic analysis using only
    message metadata and text structure (no sentiment/toxicity models).
    
    Args:
        df: DataFrame with columns: timestamp, sender, text, is_media
        
    Returns:
        {
            "global": {
                "total_messages": int,
                "days_active": int,
                "date_range_start": str (ISO),
                "date_range_end": str (ISO),
                "unique_senders": int,
            },
            "per_sender": {
                "<sender_name>": {
                    "message_count": int,
                    "word_count": int,
                    "avg_words_per_message": float,
                    "initiation_count": int,
                    "median_response_time_seconds": float,
                    "emoji_stats": {
                        "total": int,
                        "positive": int,
                        "romantic": int,
                        "playful": int,
                        "neutral": int,
                        "negative": int,
                    }
                }
            }
        }
    """
    logger.info(f"Extracting structural features from {len(df)} messages")
    
    if len(df) == 0:
        logger.warning("Empty dataframe provided")
        return _empty_features()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Global metrics
    global_metrics = {
        "total_messages": len(df),
        "days_active": _calculate_days_active(df),
        "date_range_start": df['timestamp'].min().isoformat(),
        "date_range_end": df['timestamp'].max().isoformat(),
        "unique_senders": df['sender'].nunique(),
    }
    
    # Per-sender metrics
    per_sender = {}
    senders = df['sender'].unique()
    
    for sender in senders:
        sender_df = df[df['sender'] == sender]
        
        per_sender[sender] = {
            "message_count": len(sender_df),
            "word_count": _calculate_word_count(sender_df),
            "avg_words_per_message": _calculate_avg_words(sender_df),
            "initiation_count": _calculate_initiations(df, sender),
            "median_response_time_seconds": _calculate_median_response_time(df, sender),
            "emoji_stats": _categorize_emojis(sender_df),
        }
    
    result = {
        "global": global_metrics,
        "per_sender": per_sender,
    }
    
    logger.info(f"Extracted features for {len(per_sender)} senders")
    return result


def _calculate_days_active(df: pd.DataFrame) -> int:
    """Calculate number of days between first and last message."""
    time_span = df['timestamp'].max() - df['timestamp'].min()
    return max(1, time_span.days + 1)  # +1 to include both start and end day


def _calculate_word_count(sender_df: pd.DataFrame) -> int:
    """Calculate total word count for a sender (excluding media messages)."""
    text_messages = sender_df[~sender_df['is_media']]['text']
    total_words = text_messages.str.split().str.len().sum()
    return int(total_words) if not pd.isna(total_words) else 0


def _calculate_avg_words(sender_df: pd.DataFrame) -> float:
    """Calculate average words per message for a sender."""
    text_messages = sender_df[~sender_df['is_media']]
    if len(text_messages) == 0:
        return 0.0
    
    word_counts = text_messages['text'].str.split().str.len()
    avg = word_counts.mean()
    return float(avg) if not pd.isna(avg) else 0.0


def _calculate_initiations(df: pd.DataFrame, sender: str) -> int:
    """
    Calculate conversation initiations for a sender.
    
    An initiation is defined as the first message after a silence period
    of at least 2 hours (7200 seconds).
    
    Args:
        df: Full message dataframe (sorted by timestamp)
        sender: Sender name to analyze
        
    Returns:
        Number of conversation initiations
    """
    SILENCE_THRESHOLD = timedelta(hours=2)
    
    initiations = 0
    prev_timestamp = None
    prev_sender = None
    
    for idx, row in df.iterrows():
        curr_sender = row['sender']
        curr_timestamp = row['timestamp']
        
        # First message overall is an initiation
        if prev_timestamp is None:
            if curr_sender == sender:
                initiations += 1
            prev_timestamp = curr_timestamp
            prev_sender = curr_sender
            continue
        
        # Check if this is an initiation
        time_gap = curr_timestamp - prev_timestamp
        
        if time_gap >= SILENCE_THRESHOLD and curr_sender == sender:
            initiations += 1
        
        prev_timestamp = curr_timestamp
        prev_sender = curr_sender
    
    return initiations


def _calculate_median_response_time(df: pd.DataFrame, sender: str) -> float:
    """
    Calculate median response time for a sender.
    
    Response time is the gap between a message from another sender and
    this sender's reply. Gaps > 2 hours are excluded (new conversation).
    
    Args:
        df: Full message dataframe (sorted by timestamp)
        sender: Sender name to analyze
        
    Returns:
        Median response time in seconds (or 0.0 if no valid responses)
    """
    CONVERSATION_BREAK = timedelta(hours=2)
    
    response_times = []
    prev_timestamp = None
    prev_sender = None
    
    for idx, row in df.iterrows():
        curr_sender = row['sender']
        curr_timestamp = row['timestamp']
        
        if prev_timestamp is None:
            prev_timestamp = curr_timestamp
            prev_sender = curr_sender
            continue
        
        # Check if this is a response from the target sender
        if curr_sender == sender and prev_sender != sender:
            time_gap = curr_timestamp - prev_timestamp
            
            # Exclude conversation breaks
            if time_gap < CONVERSATION_BREAK:
                response_times.append(time_gap.total_seconds())
        
        prev_timestamp = curr_timestamp
        prev_sender = curr_sender
    
    if len(response_times) == 0:
        return 0.0
    
    return float(np.median(response_times))


def _categorize_emojis(sender_df: pd.DataFrame) -> Dict[str, int]:
    """
    Categorize emojis used by a sender based on sentiment lexicon.
    
    Categories (based on emoji score in config.EMOJI_LEXICON):
    - Romantic: score >= 0.9
    - Positive: 0.5 <= score < 0.9
    - Playful: 0.2 <= score < 0.5
    - Neutral: -0.2 < score < 0.2
    - Negative: score <= -0.2
    
    Args:
        sender_df: Messages from a single sender
        
    Returns:
        Dictionary with emoji counts per category
    """
    emoji_counts = {
        "total": 0,
        "romantic": 0,
        "positive": 0,
        "playful": 0,
        "neutral": 0,
        "negative": 0,
    }
    
    # Extract all emojis from text messages
    text_messages = sender_df[~sender_df['is_media']]['text']
    
    for text in text_messages:
        if pd.isna(text):
            continue
        
        # Extract emojis using emoji library
        emojis_in_text = [char for char in text if char in emoji.EMOJI_DATA]
        
        for em in emojis_in_text:
            emoji_counts["total"] += 1
            
            # Get score from lexicon (default to 0.0 if not found)
            score = config.EMOJI_LEXICON.get(em, 0.0)
            
            # Categorize based on score
            if score >= 0.9:
                emoji_counts["romantic"] += 1
            elif score >= 0.5:
                emoji_counts["positive"] += 1
            elif score >= 0.2:
                emoji_counts["playful"] += 1
            elif score > -0.2:
                emoji_counts["neutral"] += 1
            else:
                emoji_counts["negative"] += 1
    
    return emoji_counts


def _empty_features() -> Dict[str, Any]:
    """Return empty feature structure for edge cases."""
    return {
        "global": {
            "total_messages": 0,
            "days_active": 0,
            "date_range_start": None,
            "date_range_end": None,
            "unique_senders": 0,
        },
        "per_sender": {},
    }


if __name__ == "__main__":
    # Test with sample data
    from .parser import WhatsAppParser
    from pathlib import Path
    
    sample_path = Path("sample_data/sample_chat.txt")
    
    if sample_path.exists():
        print("Testing chatstats module...")
        parser = WhatsAppParser()
        df = parser.parse_file(str(sample_path))
        
        features = extract_structural_features(df)
        
        print(f"\nGlobal Metrics:")
        for key, val in features["global"].items():
            print(f"  {key}: {val}")
        
        print(f"\nPer-Sender Metrics:")
        for sender, metrics in features["per_sender"].items():
            print(f"\n  {sender}:")
            for key, val in metrics.items():
                if key == "emoji_stats":
                    print(f"    emoji_stats:")
                    for emoji_key, emoji_val in val.items():
                        print(f"      {emoji_key}: {emoji_val}")
                else:
                    print(f"    {key}: {val}")
    else:
        print(f"Sample data not found at {sample_path}")
