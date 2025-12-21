"""
Report generation functions for ChatREL v4
Creates visualization-ready report payloads with chart data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime


def generate_chart_messages_over_time(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Generate messages over time chart data."""
    if len(df) == 0:
        return None
    
    # Group by date
    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp'].dt.date
    daily = df_copy.groupby('date').size().reset_index(name='count')
    
    return {
        "type": "line",
        "labels": [str(d) for d in daily['date']],
        "datasets": [{
            "label": "Messages",
            "data": daily['count'].tolist()
        }]
    }


def generate_chart_messages_by_sender(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate messages by sender chart data."""
    sender_data = features.get("per_sender", {})
    if not sender_data:
        return None
    
    senders = list(sender_data.keys())
    counts = [data["message_count"] for data in sender_data.values()]
    
    return {
        "type": "bar",
        "labels": senders,
        "datasets": [{
            "label": "Messages",
            "data": counts
        }]
    }


def generate_chart_words_by_sender(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate words by sender chart data."""
    sender_data = features.get("per_sender", {})
    if not sender_data:
        return None
    
    senders = list(sender_data.keys())
    words = [data["word_count"] for data in sender_data.values()]
    
    return {
        "type": "bar",
        "labels": senders,
        "datasets": [{
            "label": "Words",
            "data": words
        }]
    }


def generate_chart_response_time_by_sender(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate response time by sender chart data (in minutes)."""
    sender_data = features.get("per_sender", {})
    if not sender_data:
        return None
    
    senders = list(sender_data.keys())
    # Convert seconds to minutes
    times = [data["median_response_time_seconds"] / 60.0 for data in sender_data.values()]
    
    return {
        "type": "bar",
        "labels": senders,
        "datasets": [{
            "label": "Response Time (minutes)",
            "data": [round(t, 1) for t in times]
        }]
    }


def generate_chart_emoji_by_sender(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate total emoji count by sender chart data."""
    sender_data = features.get("per_sender", {})
    if not sender_data:
        return None
    
    senders = list(sender_data.keys())
    emoji_counts = [data["emoji_stats"]["total"] for data in sender_data.values()]
    
    return {
        "type": "bar",
        "labels": senders,
        "datasets": [{
            "label": "Emojis",
            "data": emoji_counts
        }]
    }


def generate_chart_emoji_categories(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate stacked emoji categories chart data."""
    sender_data = features.get("per_sender", {})
    if not sender_data:
        return None
    
    senders = list(sender_data.keys())
    categories = ["romantic", "positive", "playful", "neutral", "negative"]
    
    datasets = []
    for category in categories:
        data = [sender_data[s]["emoji_stats"][category] for s in senders]
        datasets.append({
            "label": category.title(),
            "data": data
        })
    
    return {
        "type": "stacked_bar",
        "labels": senders,
        "datasets": datasets
    }


def generate_chart_sentiment_over_time(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Generate sentiment over time chart data (NLP mode only)."""
    if len(df) == 0 or 'combined_sentiment' not in df.columns:
        return None
    
    # Filter valid sentiment data
    valid = df[df['combined_sentiment'].notna()].copy()
    if len(valid) == 0:
        return None
    
    # Group by date and calculate daily average sentiment
    valid['date'] = valid['timestamp'].dt.date
    daily_sentiment = valid.groupby('date')['combined_sentiment'].mean().reset_index()
    
    return {
        "type": "line",
        "labels": [str(d) for d in daily_sentiment['date']],
        "datasets": [{
            "label": "Average Sentiment",
            "data": [round(float(s), 3) for s in daily_sentiment['combined_sentiment']]
        }]
    }


def generate_report_payload(
    df: pd.DataFrame,
    structural_metrics: Dict[str, Any],
    nlp_metrics: Optional[Dict[str, Any]],
    scores: Dict[str, Any],
    relationship_type: Dict[str, Any],
    mode: str,
    sender_mapping: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate complete report payload with summary, scores, and chart data.
    
    This is the standardized report structure for frontend consumption.
    """
    # Extract summary info
    total_msgs = structural_metrics["global"]["total_messages"]
    days_active = structural_metrics["global"]["days_active"]
    start_date = structural_metrics["global"]["date_range_start"]
    end_date = structural_metrics["global"]["date_range_end"]
    
    msgs_per_day = round(total_msgs / max(1, days_active), 2)
    
    # Determine dominant sender
    sender_data = structural_metrics.get("per_sender", {})
    if len(sender_data) >= 2:
        sorted_senders = sorted(
            sender_data.items(),
            key=lambda x: x[1]["message_count"],
            reverse=True
        )
        if sorted_senders[0][1]["message_count"] / total_msgs > 0.6:
            dominant_sender = sorted_senders[0][0]
        else:
            dominant_sender = None
    else:
        dominant_sender = list(sender_data.keys())[0] if sender_data else None
    
    # Build chart data
    charts = {
        "messages_over_time": generate_chart_messages_over_time(df),
        "messages_by_sender": generate_chart_messages_by_sender(structural_metrics),
        "words_by_sender": generate_chart_words_by_sender(structural_metrics),
        "response_time_by_sender": generate_chart_response_time_by_sender(structural_metrics),
        "emoji_by_sender": generate_chart_emoji_by_sender(structural_metrics),
        "emoji_categories": generate_chart_emoji_categories(structural_metrics),
    }
    
    # Add NLP charts if available
    if mode == "formula_plus_nlp" and nlp_metrics:
        charts["sentiment_over_time"] = generate_chart_sentiment_over_time(df)
    else:
        charts["sentiment_over_time"] = None
    
    # Build report
    report = {
        "mode": mode,
        "summary": {
            "total_messages": total_msgs,
            "days_active": days_active,
            "start_date": start_date,
            "end_date": end_date,
            "messages_per_day": msgs_per_day,
            "dominant_sender": dominant_sender,
            "relationship_type": relationship_type["type"],
            "relationship_confidence": relationship_type["confidence"],
        },
        "scores": scores,
        "type_prediction": relationship_type,
        "structural_metrics": structural_metrics,
        "nlp_metrics": nlp_metrics,
        "charts": charts,
        "metadata": {
            "anonymized": sender_mapping is not None,
            "sender_mapping": sender_mapping,
            "senders": list(sender_data.keys()) if not sender_mapping else list(sender_mapping.values()),
        }
    }
    
    # Generate natural language insights
    from .insight_engine import generate_insights
    report["insights"] = generate_insights(report, mode, max_examples=3)
    
    return report
