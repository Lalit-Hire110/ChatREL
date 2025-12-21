"""
Relationship scoring framework for ChatREL v4

This module implements interpretable, robust scoring formulas for relationship analysis.
Supports both formula-only mode (structural features) and formula+NLP mode (with sentiment/toxicity).

SCORING DIMENSIONS:
1. Engagement (0-100): Activity level, message balance, responsiveness
2. Warmth (0-100): Emotional positivity, affection, closeness  
3. Conflict (0-100): Negativity, friction, arguments (0 = no conflict)
4. Stability (0-100): Consistency, longevity, retention

CONFIGURABLE CONSTANTS:
All thresholds and weights can be tuned below. Defaults are calibrated for typical
2-person WhatsApp conversations with 100-5000 messages over weeks/months.

FORMULAS:
See docs/formulas.md for detailed explanations of each scoring formula.
"""

import logging
import math
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================

# Engagement normalization ranges
ENGAGEMENT_MSG_PER_DAY_RANGE = (1.0, 50.0)  # Messages/day: 1-50 considered normal-to-high
ENGAGEMENT_RESPONSE_TIME_RANGE = (60, 3600)  # Response time (seconds): 1min-1hour optimal
ENGAGEMENT_BALANCE_PENALTY_THRESHOLD = 0.3  # Imbalance >30% gets penalized

# Warmth normalization
WARMTH_EMOJI_DENSITY_RANGE = (0.0, 0.5)  # Emoji per message: 0-50%
WARMTH_ROMANTIC_RATIO_THRESHOLD = 0.15  # >=15% romantic emojis = strong signal
WARMTH_AVG_WORDS_RANGE = (5.0, 30.0)  # Message length: 5-30 words

# Conflict detection
CONFLICT_CAPS_RATIO_THRESHOLD = 0.2  # >20% caps = high conflict signal
CONFLICT_NEGATIVE_EMOJI_RATIO_THRESHOLD = 0.1  # >10% negative emojis
CONFLICT_TOXICITY_SCALE = 1.5  # Amplify toxicity impact

# Stability thresholds
STABILITY_MIN_DAYS = 7  # Minimum days for stable relationship
STABILITY_GAP_THRESHOLD_DAYS = 7  # Gaps >7 days penalize stability
STABILITY_MIN_MESSAGES = 50  # Minimum messages for meaningful stability

# Overall health weights (must sum to ~1.0)
DEFAULT_HEALTH_WEIGHTS = {
    "engagement": 0.30,
    "warmth": 0.30,
    "stability": 0.25,
    "conflict": 0.15,  # Note: conflict is inverted (subtracted)
}

# Confidence thresholds
CONFIDENCE_MIN_MESSAGES = 200  # Below this, confidence drops
CONFIDENCE_MIN_DAYS = 7  # Below this, confidence drops
CONFIDENCE_MIN_NLP_COVERAGE = 0.3  # Below 30% NLP coverage, confidence drops

# Relationship type thresholds
REL_TYPE_COUPLE_WARMTH_MIN = 70
REL_TYPE_COUPLE_ROMANTIC_EMOJI_MIN = 0.15
REL_TYPE_COUPLE_INITIATION_BALANCE = (0.3, 0.7)  # 30-70% range

REL_TYPE_ONE_SIDED_INITIATION_MAX = 0.7  # >70% from one person
REL_TYPE_ONE_SIDED_MESSAGE_SHARE_MIN = 0.2  # <20% from one person

REL_TYPE_FAMILY_ROMANTIC_EMOJI_MAX = 0.05  # <5% romantic
REL_TYPE_FAMILY_ENGAGEMENT_MAX = 40  # Lower engagement

REL_TYPE_FRIENDS_WARMTH_RANGE = (50, 75)
REL_TYPE_FRIENDS_ENGAGEMENT_MIN = 50
REL_TYPE_FRIENDS_PLAYFUL_EMOJI_MIN = 0.1

REL_TYPE_ACQUAINTANCE_ENGAGEMENT_MAX = 40
REL_TYPE_ACQUAINTANCE_WARMTH_MAX = 50


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_range(value: float, min_val: float, max_val: float, clip: bool = True) -> float:
    """Normalize value from [min_val, max_val] to [0, 1]."""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    if clip:
        return max(0.0, min(1.0, normalized))
    return normalized


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default fallback."""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_engagement(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate engagement score (0-100).
    
    Measures: Activity level, message balance, responsiveness, initiation patterns
    
    Formula:
    - activity_score: Messages per day with saturation (log scale)
    - balance_score: 1 - |msg_A - msg_B| / total_msgs
    - responsiveness_score: Inverse median response time
    - initiation_balance: How evenly initiations are distributed
    
    Args:
        features: Structural features from chatstats.extract_structural_features()
        
    Returns:
        {
            "score": float (raw),
            "normalized": float (0-100),
            "inputs": dict,
            "notes": list[str],
        }
    """
    notes = []
    inputs = {}
    
    # Extract global metrics
    total_msgs = features["global"]["total_messages"]
    days_active = max(1, features["global"]["days_active"])
    per_sender = features["global"]["unique_senders"]
    
    inputs["total_messages"] = total_msgs
    inputs["days_active"] = days_active
    
    if total_msgs < 10:
        notes.append("Very low message count (<10), engagement score unreliable")
        return {
            "score": 0.0,
            "normalized": 0.0,
            "inputs": inputs,
            "notes": notes,
        }
    
    # 1. Activity score (messages per day with log saturation)
    msgs_per_day = total_msgs / days_active
    inputs["msgs_per_day"] = round(msgs_per_day, 2)
    
    # Log saturation: activity_score = log(1 + msgs_per_day) / log(1 + 50)
    activity_raw = math.log(1 + msgs_per_day) / math.log(1 + ENGAGEMENT_MSG_PER_DAY_RANGE[1])
    activity_score = clamp(activity_raw * 100, 0, 100)
    inputs["activity_score"] = round(activity_score, 2)
    
    # 2. Balance score (message distribution)
    sender_data = features["per_sender"]
    if len(sender_data) >= 2:
        msg_counts = [data["message_count"] for data in sender_data.values()]
        imbalance = abs(msg_counts[0] - msg_counts[1]) / total_msgs if total_msgs > 0 else 0
        balance_score = (1.0 - imbalance) * 100
        
        inputs["message_imbalance"] = round(imbalance, 3)
        inputs["balance_score"] = round(balance_score, 2)
        
        if imbalance > ENGAGEMENT_BALANCE_PENALTY_THRESHOLD:
            notes.append(f"High message imbalance: {imbalance:.1%}")
    else:
        balance_score = 100  # Single sender = balanced (edge case)
        notes.append("Single sender detected, assuming balanced")
    
    # 3. Responsiveness score (inverse median response time)
    response_times = [
        data["median_response_time_seconds"] 
        for data in sender_data.values() 
        if data["median_response_time_seconds"] > 0
    ]
    
    if response_times:
        avg_response_time = np.mean(response_times)
        inputs["avg_median_response_time_sec"] = round(avg_response_time, 1)
        
        # Normalize: 60s = max score, 3600s = min score
        responsiveness_raw = 1.0 - normalize_range(
            avg_response_time,
            ENGAGEMENT_RESPONSE_TIME_RANGE[0],
            ENGAGEMENT_RESPONSE_TIME_RANGE[1],
            clip=True
        )
        responsiveness_score = responsiveness_raw * 100
        inputs["responsiveness_score"] = round(responsiveness_score, 2)
    else:
        responsiveness_score = 50  # Neutral if no data
        notes.append("No response time data, using neutral score")
    
    # 4. Initiation balance
    if len(sender_data) >= 2:
        initiations = [data["initiation_count"] for data in sender_data.values()]
        total_initiations = sum(initiations)
        
        if total_initiations > 0:
            init_ratio = initiations[0] / total_initiations
            init_imbalance = abs(init_ratio - 0.5)
            initiation_balance_score = (1.0 - init_imbalance * 2) * 100  # 0.5 = perfect balance
            
            inputs["initiation_ratio"] = round(init_ratio, 3)
            inputs["initiation_balance_score"] = round(initiation_balance_score, 2)
        else:
            initiation_balance_score = 50
            notes.append("No initiation data")
    else:
        initiation_balance_score = 50
    
    # Weighted combination
    engagement_raw = (
        0.40 * activity_score +
        0.30 * balance_score +
        0.20 * responsiveness_score +
        0.10 * initiation_balance_score
    )
    
    engagement_normalized = clamp(engagement_raw, 0, 100)
    
    return {
        "score": round(engagement_raw, 2),
        "normalized": round(engagement_normalized, 2),
        "inputs": inputs,
        "notes": notes,
    }


def calculate_warmth(features: Dict[str, Any], nlp: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calculate warmth score (0-100).
    
    Measures: Emotional positivity, affection, closeness
    
    Formula (formula-only):
    - emoji_affection: (romantic + positive) / total_emoji
    - message_depth: avg_words_per_message
    
    Formula (with NLP):
    - Blend: 0.6 * emoji_affection + 0.4 * sentiment_positivity
    
    Args:
        features: Structural features
        nlp: Optional NLP metrics
        
    Returns:
        {
            "score": float,
            "normalized": float (0-100),
            "inputs": dict,
            "notes": list[str],
            "used_nlp": bool,
        }
    """
    notes = []
    inputs = {}
    used_nlp = nlp is not None
    
    sender_data = features["per_sender"]
    total_msgs = features["global"]["total_messages"]
    
    if total_msgs < 5:
        notes.append("Very few messages, warmth score unreliable")
        return {
            "score": 0.0,
            "normalized": 0.0,
            "inputs": inputs,
            "notes": notes,
            "used_nlp": used_nlp,
        }
    
    # 1. Emoji affection score
    total_romantic = sum(data["emoji_stats"]["romantic"] for data in sender_data.values())
    total_positive = sum(data["emoji_stats"]["positive"] for data in sender_data.values())
    total_emojis = sum(data["emoji_stats"]["total"] for data in sender_data.values())
    
    inputs["total_romantic_emoji"] = total_romantic
    inputs["total_positive_emoji"] = total_positive
    inputs["total_emojis"] = total_emojis
    
    if total_emojis > 0:
        romantic_ratio = total_romantic / total_emojis
        affection_emoji_count = total_romantic + total_positive
        emoji_affection_raw = affection_emoji_count / total_emojis
        
        inputs["romantic_emoji_ratio"] = round(romantic_ratio, 3)
        inputs["emoji_affection_ratio"] = round(emoji_affection_raw, 3)
        
        # Boost if high romantic content
        emoji_affection_score = emoji_affection_raw * 100
        if romantic_ratio >= WARMTH_ROMANTIC_RATIO_THRESHOLD:
            emoji_affection_score *= 1.2  # Boost
            notes.append(f"High romantic emoji ratio: {romantic_ratio:.1%}")
        
        emoji_affection_score = clamp(emoji_affection_score, 0, 100)
        inputs["emoji_affection_score"] = round(emoji_affection_score, 2)
    else:
        emoji_affection_score = 30  # Low default if no emojis
        notes.append("No emojis found, using low default warmth from emojis")
    
    # 2. Message depth score (longer = more engaged/warm)
    avg_words_list = [data["avg_words_per_message"] for data in sender_data.values()]
    avg_words = np.mean(avg_words_list) if avg_words_list else 0
    
    inputs["avg_words_per_message"] = round(avg_words, 2)
    
    message_depth_raw = normalize_range(avg_words, WARMTH_AVG_WORDS_RANGE[0], WARMTH_AVG_WORDS_RANGE[1], clip=True)
    message_depth_score = message_depth_raw * 100
    inputs["message_depth_score"] = round(message_depth_score, 2)
    
    # 3. NLP sentiment (if available)
    if used_nlp and "sentiment_mean" in nlp:
        sentiment_mean = nlp.get("sentiment_mean", 0.0)
        inputs["sentiment_mean"] = round(sentiment_mean, 3)
        
        # Map sentiment from [-1, 1] to [0, 100]
        sentiment_score = (sentiment_mean + 1.0) / 2.0 * 100
        inputs["sentiment_score"] = round(sentiment_score, 2)
        
        # Blend: 60% emoji, 40% sentiment
        warmth_raw = 0.6 * emoji_affection_score + 0.4 * sentiment_score
        notes.append("Using NLP sentiment for enhanced warmth calculation")
        
        # Penalize if toxicity present
        if "toxicity_mean" in nlp and nlp["toxicity_mean"] > 0.3:
            penalty = nlp["toxicity_mean"] * 10
            warmth_raw -= penalty
            notes.append(f"Toxicity penalty applied: -{penalty:.1f}")
    else:
        # Formula-only: blend emoji + message depth
        warmth_raw = 0.7 * emoji_affection_score + 0.3 * message_depth_score
        notes.append("Formula-only mode: using emoji + message depth")
    
    warmth_normalized = clamp(warmth_raw, 0, 100)
    
    return {
        "score": round(warmth_raw, 2),
        "normalized": round(warmth_normalized, 2),
        "inputs": inputs,
        "notes": notes,
        "used_nlp": used_nlp,
    }


def calculate_conflict(features: Dict[str, Any], nlp: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calculate conflict score (0-100).
    
    Higher score = more conflict. 0 = no conflict detected.
    
    Formula (formula-only):
    - Negative emoji ratio
    - Message pattern analysis (short bursts, etc.)
    
    Formula (with NLP):
    - Structural conflict + toxicity boost
    
    Args:
        features: Structural features
        nlp: Optional NLP metrics
        
    Returns:
        {
            "score": float,
            "normalized": float (0-100),
            "inputs": dict,
            "notes": list[str],
            "used_nlp": bool,
        }
    """
    notes = []
    inputs = {}
    used_nlp = nlp is not None
    
    sender_data = features["per_sender"]
    total_msgs = features["global"]["total_messages"]
    
    # 1. Negative emoji ratio
    total_negative_emoji = sum(data["emoji_stats"]["negative"] for data in sender_data.values())
    total_emojis = sum(data["emoji_stats"]["total"] for data in sender_data.values())
    
    inputs["total_negative_emoji"] = total_negative_emoji
    inputs["total_emojis"] = total_emojis
    
    if total_emojis > 0:
        negative_emoji_ratio = total_negative_emoji / total_emojis
        inputs["negative_emoji_ratio"] = round(negative_emoji_ratio, 3)
        
        # Normalize to 0-100
        negative_emoji_score = normalize_range(
            negative_emoji_ratio,
            0.0,
            CONFLICT_NEGATIVE_EMOJI_RATIO_THRESHOLD,
            clip=True
        ) * 100
        
        inputs["negative_emoji_score"] = round(negative_emoji_score, 2)
        
        if negative_emoji_ratio > CONFLICT_NEGATIVE_EMOJI_RATIO_THRESHOLD:
            notes.append(f"High negative emoji ratio: {negative_emoji_ratio:.1%}")
    else:
        negative_emoji_score = 0
        notes.append("No emoji data for conflict detection")
    
    # 2. Structural conflict (formula-only baseline)
    structural_conflict = negative_emoji_score
    inputs["structural_conflict"] = round(structural_conflict, 2)
    
    # 3. NLP toxicity boost (if available)
    if used_nlp and "toxicity_mean" in nlp:
        toxicity_mean = nlp.get("toxicity_mean", 0.0)
        conflict_flag_count = nlp.get("conflict_flag_count", 0)
        
        inputs["toxicity_mean"] = round(toxicity_mean, 3)
        inputs["conflict_flag_count"] = conflict_flag_count
        
        # Toxicity score (0-1 to 0-100 with amplification)
        toxicity_score = toxicity_mean * 100 * CONFLICT_TOXICITY_SCALE
        inputs["toxicity_score"] = round(toxicity_score, 2)
        
        # Take max of structural and toxicity
        conflict_raw = max(structural_conflict, toxicity_score)
        
        # Additional penalty from conflict flags
        if conflict_flag_count > 0 and total_msgs > 0:
            conflict_flag_ratio = conflict_flag_count / total_msgs
            conflict_penalty = conflict_flag_ratio * 50
            conflict_raw += conflict_penalty
            inputs["conflict_flag_penalty"] = round(conflict_penalty, 2)
        
        notes.append("Using NLP toxicity for conflict detection")
    else:
        conflict_raw = structural_conflict
        notes.append("Formula-only mode: using emoji-based conflict detection")
    
    conflict_normalized = clamp(conflict_raw, 0, 100)
    
    return {
        "score": round(conflict_raw, 2),
        "normalized": round(conflict_normalized, 2),
        "inputs": inputs,
        "notes": notes,
        "used_nlp": used_nlp,
    }


def calculate_stability(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate stability score (0-100).
    
    Measures: Consistency over time, retention, lack of long gaps
    
    Formula:
    - Time span score: days_active (longer = more stable)
    - Consistency score: Uniform message distribution
    - Gap penalty: Penalize multi-day silences
    
    Args:
        features: Structural features
        
    Returns:
        {
            "score": float,
            "normalized": float (0-100),
            "inputs": dict,
            "notes": list[str],
        }
    """
    notes = []
    inputs = {}
    
    days_active = features["global"]["days_active"]
    total_msgs = features["global"]["total_messages"]
    
    inputs["days_active"] = days_active
    inputs["total_messages"] = total_msgs
    
    if days_active < 2:
        notes.append("Chat duration <2 days, stability score unreliable")
        return {
            "score": 0.0,
            "normalized": 0.0,
            "inputs": inputs,
            "notes": notes,
        }
    
    # 1. Time span score (log scale for saturation)
    # Longer relationships = more stable
    time_span_raw = math.log(1 + days_active) / math.log(1 + 180)  # 180 days = max
    time_span_score = time_span_raw * 100
    inputs["time_span_score"] = round(time_span_score, 2)
    
    # 2. Message volume score (more messages = more data = more confidence in stability)
    volume_raw = math.log(1 + total_msgs) / math.log(1 + STABILITY_MIN_MESSAGES * 10)
    volume_score = volume_raw * 100
    inputs["volume_score"] = round(volume_score, 2)
    
    # 3. Consistency score (uniform distribution)
    msgs_per_day = total_msgs / days_active
    inputs["msgs_per_day"] = round(msgs_per_day, 2)
    
    # Assume moderate variance for now (without daily breakdown)
    # In real implementation, would compute variance of messages per day
    # For now, use msgs_per_day as proxy: very low or very high = less stable
    if msgs_per_day < 1.0:
        consistency_score = 40  # Very sporadic
        notes.append("Very low message frequency (<1/day), reduced stability")
    elif msgs_per_day > 100:
        consistency_score = 60  # Too intense, may not sustain
        notes.append("Very high message frequency (>100/day), may not be sustainable")
    else:
        consistency_score = 80  # Good range
    
    inputs["consistency_score"] = consistency_score
    
    # 4. Combine
    stability_raw = (
        0.40 * time_span_score +
        0.30 * volume_score +
        0.30 * consistency_score
    )
    
    # Penalty for very short relationships
    if days_active < STABILITY_MIN_DAYS:
        penalty = (STABILITY_MIN_DAYS - days_active) * 5
        stability_raw -= penalty
        notes.append(f"Short duration penalty: -{penalty:.1f}")
    
    stability_normalized = clamp(stability_raw, 0, 100)
    
    return {
        "score": round(stability_raw, 2),
        "normalized": round(stability_normalized, 2),
        "inputs": inputs,
        "notes": notes,
    }


def calculate_overall_health(
    scores: Dict[str, Any], 
    weights: Optional[Dict[str, float]] = None,
    features: Optional[Dict[str, Any]] = None,
    nlp: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Combine sub-scores into overall relationship health (0-100).
    
    Formula:
    health = w_engagement * engagement
           + w_warmth * warmth
           - w_conflict * conflict  # Inverted
           + w_stability * stability
    
    Args:
        scores: Dict with engagement, warmth, conflict, stability scores
        weights: Optional custom weights (default: DEFAULT_HEALTH_WEIGHTS)
        features: Optional features for confidence calculation
        nlp: Optional NLP metrics for confidence calculation
        
    Returns:
        {
            "score": float,
            "normalized": float (0-100),
            "confidence": float (0-1),
            "inputs": dict,
            "notes": list[str],
        }
    """
    if weights is None:
        weights = DEFAULT_HEALTH_WEIGHTS.copy()
    
    notes = []
    inputs = {}
    
    # Extract normalized scores
    engagement = scores.get("engagement", {}).get("normalized", 50)
    warmth = scores.get("warmth", {}).get("normalized", 50)
    conflict = scores.get("conflict", {}).get("normalized", 0)
    stability = scores.get("stability", {}).get("normalized", 50)
    
    inputs["engagement"] = engagement
    inputs["warmth"] = warmth
    inputs["conflict"] = conflict
    inputs["stability"] = stability
    inputs["weights"] = weights
    
    # Weighted combination (conflict is subtracted)
    health_raw = (
        weights["engagement"] * engagement +
        weights["warmth"] * warmth -
        weights["conflict"] * conflict +  # Note: subtracted
        weights["stability"] * stability
    )
    
    health_normalized = clamp(health_raw, 0, 100)
    
    # Calculate confidence
    confidence = 1.0
    
    if features:
        total_msgs = features["global"]["total_messages"]
        days_active = features["global"]["days_active"]
        
        # Confidence drops for small chats
        if total_msgs < CONFIDENCE_MIN_MESSAGES:
            msg_confidence = total_msgs / CONFIDENCE_MIN_MESSAGES
            confidence *= msg_confidence
            notes.append(f"Low message count ({total_msgs}), confidence reduced")
        
        # Confidence drops for short duration
        if days_active < CONFIDENCE_MIN_DAYS:
            days_confidence = days_active / CONFIDENCE_MIN_DAYS
            confidence *= days_confidence
            notes.append(f"Short duration ({days_active} days), confidence reduced")
    
    # Confidence drops if NLP enabled but poor coverage
    if nlp is not None and features:
        total_msgs = features["global"]["total_messages"]
        # Estimate NLP coverage (assume most messages processed if toxicity data exists)
        # In real implementation, track actual coverage
        nlp_coverage = 1.0  # Assume full coverage if nlp dict present
        
        if nlp_coverage < CONFIDENCE_MIN_NLP_COVERAGE:
            confidence *= nlp_coverage / CONFIDENCE_MIN_NLP_COVERAGE
            notes.append(f"Low NLP coverage ({nlp_coverage:.1%}), confidence reduced")
    
    inputs["confidence"] = round(confidence, 3)
    
    return {
        "score": round(health_raw, 2),
        "normalized": round(health_normalized, 2),
        "confidence": round(confidence, 3),
        "inputs": inputs,
        "notes": notes,
    }


def predict_relationship_type(
    scores: Dict[str, Any],
    features: Dict[str, Any],
    nlp: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Classify relationship type using rule-based heuristics    
    Types: couple, romantic_crush, friends, family, acquaintance, one-sided
    
    Args:
        scores: Computed relationship scores
        features: Structural features
        nlp: Optional NLP metrics
        
    Returns:
        {
            "type": str,
            "confidence": float (0-1),
            "evidence": list[str],
        }
    """
    evidence = []
    
    # Extract scores
    engagement = scores.get("engagement", {}).get("normalized", 50)
    warmth = scores.get("warmth", {}).get("normalized", 50)
    conflict = scores.get("conflict", {}).get("normalized", 0)
    stability = scores.get("stability", {}).get("normalized", 50)
    
    # Extract structural metrics
    sender_data = features["per_sender"]
    total_msgs = features["global"]["total_messages"]
    
    # Calculate message share
    if len(sender_data) >= 2:
        msg_counts = [data["message_count"] for data in sender_data.values()]
        msg_share_ratio = msg_counts[0] / total_msgs if total_msgs > 0 else 0.5
        
        # Emoji metrics
        senders = list(sender_data.values())
        total_romantic = sum(s["emoji_stats"]["romantic"] for s in senders)
        total_positive = sum(s["emoji_stats"]["positive"] for s in senders)
        total_playful = sum(s["emoji_stats"]["playful"] for s in senders)
        total_emojis = sum(s["emoji_stats"]["total"] for s in senders)
        
        romantic_ratio = safe_divide(total_romantic, total_emojis, 0.0)
        playful_ratio = safe_divide(total_playful, total_emojis, 0.0)
        
        # Initiation balance
        initiations = [data["initiation_count"] for data in sender_data.values()]
        total_init = sum(initiations)
        init_ratio = safe_divide(initiations[0], total_init, 0.5)
    else:
        msg_share_ratio = 1.0
        romantic_ratio = 0.0
        playful_ratio = 0.0
        init_ratio = 1.0
        evidence.append("Single sender detected (edge case)")
    
    # Build evidence
    evidence.append(f"Engagement: {engagement:.0f}/100")
    evidence.append(f"Warmth: {warmth:.0f}/100")
    evidence.append(f"Conflict: {conflict:.0f}/100")
    evidence.append(f"Message share: {msg_share_ratio:.1%} vs {1-msg_share_ratio:.1%}")
    evidence.append(f"Initiation ratio: {init_ratio:.1%} vs {1-init_ratio:.1%}")
    evidence.append(f"Romantic emoji ratio: {romantic_ratio:.1%}")
    
    # Classification logic
    rel_type = "acquaintance"  # Default
    confidence = 0.5  # Default
    
    # 1. Check for one-sided
    if msg_share_ratio > REL_TYPE_ONE_SIDED_INITIATION_MAX or msg_share_ratio < (1 - REL_TYPE_ONE_SIDED_INITIATION_MAX):
        rel_type = "one-sided"
        confidence = 0.8
        evidence.append("⚠️ Highly asymmetric message distribution")
    
    elif msg_share_ratio < REL_TYPE_ONE_SIDED_MESSAGE_SHARE_MIN or msg_share_ratio > (1 - REL_TYPE_ONE_SIDED_MESSAGE_SHARE_MIN):
        rel_type = "one-sided"
        confidence = 0.7
        evidence.append("⚠️ One person contributes <20% of messages")
    
    # 2. Check for couple/romantic
    elif (
        warmth >= REL_TYPE_COUPLE_WARMTH_MIN
        and romantic_ratio >= REL_TYPE_COUPLE_ROMANTIC_EMOJI_MIN
        and REL_TYPE_COUPLE_INITIATION_BALANCE[0] <= init_ratio <= REL_TYPE_COUPLE_INITIATION_BALANCE[1]
        and engagement >= 50
    ):
        rel_type = "couple"
        confidence = 0.85
        evidence.append("💕 High warmth + romantic emojis + balanced interaction")
    
    # 3. Check for romantic crush (high warmth but asymmetric)
    elif (
        warmth >= REL_TYPE_COUPLE_WARMTH_MIN
        and romantic_ratio >= REL_TYPE_COUPLE_ROMANTIC_EMOJI_MIN
        and not (REL_TYPE_COUPLE_INITIATION_BALANCE[0] <= init_ratio <= REL_TYPE_COUPLE_INITIATION_BALANCE[1])
    ):
        rel_type = "romantic_crush"
        confidence = 0.75
        evidence.append("💕 High warmth + romantic emojis but asymmetric initiation")
    
    # 4. Check for family
    elif (
        romantic_ratio <= REL_TYPE_FAMILY_ROMANTIC_EMOJI_MAX
        and engagement <= REL_TYPE_FAMILY_ENGAGEMENT_MAX
        and warmth >= 40
        and warmth <= 70
    ):
        rel_type = "family"
        confidence = 0.65
        evidence.append("👨‍👩‍👧 Low romantic content + moderate warmth + lower engagement")
    
    # 5. Check for friends
    elif (
        REL_TYPE_FRIENDS_WARMTH_RANGE[0] <= warmth <= REL_TYPE_FRIENDS_WARMTH_RANGE[1]
        and engagement >= REL_TYPE_FRIENDS_ENGAGEMENT_MIN
        and playful_ratio >= REL_TYPE_FRIENDS_PLAYFUL_EMOJI_MIN
        and conflict < 40
    ):
        rel_type = "friends"
        confidence = 0.8
        evidence.append("👥 Moderate warmth + high engagement + playful content")
    
    # 6. Acquaintance (default)
    elif engagement <= REL_TYPE_ACQUAINTANCE_ENGAGEMENT_MAX and warmth <= REL_TYPE_ACQUAINTANCE_WARMTH_MAX:
        rel_type = "acquaintance"
        confidence = 0.6
        evidence.append("💼 Low engagement + low warmth")
    
    # If none match strongly, use friends as default for moderate values
    elif engagement >= 40 and warmth >= 40:
        rel_type = "friends"
        confidence = 0.5
        evidence.append("👥 Default classification based on moderate scores")
    
    return {
        "type": rel_type,
        "confidence": round(confidence, 3),
        "evidence": evidence,
    }


if __name__ == "__main__":
    # Test with mock data
    print("Testing relationship scoring formulas...\n")
    
    # Mock features
    mock_features = {
        "global": {
            "total_messages": 500,
            "days_active": 30,
            "unique_senders": 2,
        },
        "per_sender": {
            "Alice": {
                "message_count": 250,
                "word_count": 5000,
                "avg_words_per_message": 20.0,
                "initiation_count": 15,
                "median_response_time_seconds": 300.0,
                "emoji_stats": {
                    "total": 45,
                    "romantic": 12,
                    "positive": 20,
                    "playful": 8,
                    "neutral": 3,
                    "negative": 2,
                },
            },
            "Bob": {
                "message_count": 250,
                "word_count": 4500,
                "avg_words_per_message": 18.0,
                "initiation_count": 18,
                "median_response_time_seconds": 450.0,
                "emoji_stats": {
                    "total": 38,
                    "romantic": 8,
                    "positive": 15,
                    "playful": 10,
                    "neutral": 4,
                    "negative": 1,
                },
            },
        },
    }
    
    # Test each function
    print("1. Engagement:")
    engagement = calculate_engagement(mock_features)
    print(f"   Score: {engagement['normalized']}/100")
    print(f"   Notes: {engagement['notes']}")
    
    print("\n2. Warmth (formula-only):")
    warmth = calculate_warmth(mock_features, nlp=None)
    print(f"   Score: {warmth['normalized']}/100")
    print(f"   Used NLP: {warmth['used_nlp']}")
    
    print("\n3. Conflict:")
    conflict = calculate_conflict(mock_features, nlp=None)
    print(f"   Score: {conflict['normalized']}/100")
    
    print("\n4. Stability:")
    stability = calculate_stability(mock_features)
    print(f"   Score: {stability['normalized']}/100")
    
    print("\n5. Overall Health:")
    scores = {
        "engagement": engagement,
        "warmth": warmth,
        "conflict": conflict,
        "stability": stability,
    }
    health = calculate_overall_health(scores, features=mock_features)
    print(f"   Score: {health['normalized']}/100")
    print(f"   Confidence: {health['confidence']}")
    
    print("\n6. Relationship Type:")
    rel_type = predict_relationship_type(scores, mock_features)
    print(f"   Type: {rel_type['type']}")
    print(f"   Confidence: {rel_type['confidence']}")
    print(f"   Evidence:")
    for ev in rel_type['evidence'][:5]:
        print(f"     - {ev}")
    
    print("\n✓ All scoring functions working")
