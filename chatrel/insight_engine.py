"""
ChatREL Insight Engine
Generates natural language insights from relationship analytics

Style: Warm, smart friend analyzing - not clinical or therapeutic
Tone: Confident but gentle, slightly witty, readable
"""

import random
from typing import Dict, Any, List, Optional


# Phrase banks for varied, natural language
SUMMARY_INTROS = [
    "This conversation looks",
    "From what we can see,",
    "Looking at the patterns,",
    "At first glance,",
]

BALANCE_PHRASES = {
    "balanced": [
        "pretty even between you two",
        "nicely balanced",
        "fairly mutual",
        "well-matched in back-and-forth",
    ],
    "slight_imbalance": [
        "mostly even, though {dominant} seems to {action} a bit more often",
        "fairly balanced, with {dominant} leading slightly",
    ],
    "imbalanced": [
        "{dominant} carries more of the conversation weight",
        "there's a noticeable lean toward {dominant}'s side",
    ],
}

WARMTH_DESCRIPTORS = {
    "high": ["solid warmth", "plenty of positive vibes", "warm and expressive", "emotionally open"],
    "medium": ["moderate warmth", "friendly but not overly effusive", "casual warmth"],
    "low": ["more reserved", "emotionally restrained", "fairly neutral"],
}

CONFLICT_DESCRIPTORS = {
    "low": ["low conflict", "mostly tension-free", "smooth communication"],
    "medium": ["some friction points", "occasional tension"],
    "high": ["noticeable conflict signals", "some tense exchanges"],
}


def generate_insights(
    report: Dict[str, Any],
    mode: str,
    max_examples: int = 3
) -> Dict[str, Any]:
    """
    Generate natural language insights from analytics report.
    
    Args:
        report: Full report dict from generate_report()
        mode: "formula_only" or "formula_plus_nlp"
        max_examples: Max message samples to include (NLP mode only)
        
    Returns:
        {
            "summary": str,
            "strengths": [str, ...],
            "risks": [str, ...],
            "suggestions": [str, ...],
            "tone": "neutral_friendly",
            "used_samples": [...]
        }
    """
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    type_pred = report.get("type_prediction", {})
    structural = report.get("structural_metrics", {})
    nlp = report.get("nlp_metrics")
    
    # Generate each section
    summary_text = _generate_summary(summary, scores, type_pred, structural, mode)
    strengths = _generate_strengths(summary, scores, structural, nlp, mode)
    risks = _generate_risks(summary, scores, structural, nlp, mode)
    suggestions = _generate_suggestions(summary, scores, risks, mode)
    
    # Optional: Sample messages (NLP mode only)
    used_samples = []
    if mode == "formula_plus_nlp" and nlp:
        used_samples = _extract_sample_messages(nlp, max_examples)
    
    return {
        "summary": summary_text,
        "strengths": strengths,
        "risks": risks,
        "suggestions": suggestions,
        "tone": "neutral_friendly",
        "used_samples": used_samples,
    }


def _generate_summary(
    summary: Dict,
    scores: Dict,
    type_pred: Dict,
    structural: Dict,
    mode: str
) -> str:
    """Generate 4-8 sentence overview."""
    parts = []
    
    # Intro + relationship type
    intro = random.choice(SUMMARY_INTROS)
    rel_type = summary.get("relationship_type", "connection").replace("_", " ")
    confidence = summary.get("relationship_confidence", 0.5)
    
    if confidence >= 0.75:
        parts.append(f"{intro} like a {rel_type} relationship — pretty clear from the patterns.")
    elif confidence >= 0.5:
        parts.append(f"{intro} like it could be a {rel_type} dynamic, though there's some nuance.")
    else:
        parts.append(f"{intro} like there are mixed signals, making the exact dynamic harder to pin down.")
    
    # Activity level
    total_msgs = summary.get("total_messages", 0)
    days = summary.get("days_active", 1)
    msgs_per_day = summary.get("messages_per_day", 0)
    
    if msgs_per_day >= 20:
        parts.append("This is a pretty active conversation — lots of messages flowing regularly.")
    elif msgs_per_day >= 5:
        parts.append("You two chat fairly consistently, keeping things going without overwhelming frequency.")
    else:
        parts.append("The conversation is more sporadic, with quieter stretches between messages.")
    
    # Balance
    dominant = summary.get("dominant_sender")
    per_sender = structural.get("per_sender", {})
    
    if not dominant:
        balance_phrase = random.choice(BALANCE_PHRASES["balanced"])
        parts.append(f"The message balance is {balance_phrase}.")
    elif len(per_sender) >= 2:
        senders = list(per_sender.keys())
        other = [s for s in senders if s != dominant][0] if len(senders) > 1 else "the other person"
        
        dominant_msgs = per_sender[dominant]["message_count"]
        total = summary.get("total_messages", 1)
        ratio = dominant_msgs / total
        
        if ratio > 0.7:
            phrase = random.choice(BALANCE_PHRASES["imbalanced"]).format(dominant=dominant)
        else:
            phrase = random.choice(BALANCE_PHRASES["slight_imbalance"]).format(
                dominant=dominant,
                action="contribute" if ratio > 0.6 else "initiate"
            )
        parts.append(phrase + ".")
    
    # Warmth & Conflict
    warmth = scores.get("warmth", {}).get("normalized", 50)
    conflict = scores.get("conflict", {}).get("normalized", 0)
    
    if warmth >= 70:
        warmth_desc = random.choice(WARMTH_DESCRIPTORS["high"])
    elif warmth >= 40:
        warmth_desc = random.choice(WARMTH_DESCRIPTORS["medium"])
    else:
        warmth_desc = random.choice(WARMTH_DESCRIPTORS["low"])
    
    if conflict <= 20:
        conflict_desc = random.choice(CONFLICT_DESCRIPTORS["low"])
    elif conflict <= 50:
        conflict_desc = random.choice(CONFLICT_DESCRIPTORS["medium"])
    else:
        conflict_desc = random.choice(CONFLICT_DESCRIPTORS["high"])
    
    parts.append(f"Warmth shows {warmth_desc}, and conflict is {conflict_desc}.")
    
    # Mode acknowledgment
    if mode == "formula_only":
        parts.append("(Note: This analysis uses message patterns only — no sentiment AI, so emotional tone is inferred from structure and emojis.)")
    
    return " ".join(parts)


def _generate_strengths(
    summary: Dict,
    scores: Dict,
    structural: Dict,
    nlp: Optional[Dict],
    mode: str
) -> List[str]:
    """Identify and celebrate positive patterns."""
    strengths = []
    
    # High engagement
    engagement = scores.get("engagement", {}).get("normalized", 0)
    if engagement >= 70:
        strengths.append("Strong engagement — you both actively participate and keep things moving.")
    
    # Balanced messaging
    dominant = summary.get("dominant_sender")
    if not dominant:
        strengths.append("Nice balance in who talks — feels collaborative rather than one-sided.")
    
    # Fast responses
    per_sender = structural.get("per_sender", {})
    response_times = [s["median_response_time_seconds"] for s in per_sender.values()]
    if response_times and max(response_times) < 600:  # Under 10 minutes
        strengths.append("Quick response times — shows attentiveness and availability.")
    
    # High warmth
    warmth = scores.get("warmth", {}).get("normalized", 0)
    if warmth >= 70:
        total_emojis = sum(s["emoji_stats"]["total"] for s in per_sender.values())
        romantic_emojis = sum(s["emoji_stats"]["romantic"] for s in per_sender.values())
        
        if romantic_emojis >= 10:
            strengths.append("Plenty of warm, affectionate signals — especially through emoji and tone.")
        else:
            strengths.append("Positive emotional tone comes through clearly in how you communicate.")
    
    # Consistency
    stability = scores.get("stability", {}).get("normalized", 0)
    if stability >= 60:
        strengths.append("Consistent communication over time — shows sustained interest and effort.")
    
    # NLP-specific
    if mode == "formula_plus_nlp" and nlp:
        sentiment = nlp.get("sentiment_mean", 0)
        if sentiment >= 0.4:
            strengths.append("Positive sentiment throughout — the vibe is generally upbeat and friendly.")
    
    # Fallback if no strengths
    if len(strengths) == 0:
        strengths.append("The conversation exists and has patterns worth exploring.")
    
    return strengths[:5]  # Cap at 5


def _generate_risks(
    summary: Dict,
    scores: Dict,
    structural: Dict,
    nlp: Optional[Dict],
    mode: str
) -> List[str]:
    """Surface potential friction areas with neutral phrasing."""
    risks = []
    
    # One-sided imbalance
    dominant = summary.get("dominant_sender")
    per_sender = structural.get("per_sender", {})
    
    if dominant and len(per_sender) >= 2:
        dominant_msgs = per_sender[dominant]["message_count"]
        total = summary.get("total_messages", 1)
        ratio = dominant_msgs / total
        
        if ratio > 0.75:
            risks.append(f"One person ({dominant}) carries most of the conversation. This might reflect different communication styles or schedules, but it could also signal uneven effort.")
    
    # Initiation imbalance
    if len(per_sender) >= 2:
        initiations = [s["initiation_count"] for s in per_sender.values()]
        total_init = sum(initiations)
        if total_init > 0:
            init_ratio = max(initiations) / total_init
            if init_ratio > 0.7:
                initiator = list(per_sender.keys())[0] if initiations[0] > initiations[1] else list(per_sender.keys())[1]
                risks.append(f"{initiator} starts most conversations. This pattern can sometimes feel one-sided over time.")
    
    # Slow responses
    response_times = [s["median_response_time_seconds"] for s in per_sender.values()]
    if response_times and max(response_times) > 7200:  # Over 2 hours
        slow_responder = list(per_sender.keys())[response_times.index(max(response_times))]
        risks.append(f"{slow_responder}'s response times are notably slower. Could be normal for their routine, or might indicate lower priority.")
    
    # Low warmth
    warmth = scores.get("warmth", {}).get("normalized", 0)
    if warmth < 40:
        risks.append("Warmth signals are relatively low. This might just mean a more reserved style, or it could suggest emotional distance.")
    
    # High conflict
    conflict = scores.get("conflict", {}).get("normalized", 0)
    if conflict > 50:
        if mode == "formula_plus_nlp":
            risks.append("Some tension or negative exchanges show up in the data. Worth noticing whether this is occasional venting or a recurring pattern.")
        else:
            risks.append("Conflict indicators (negative emojis, patterns) are present. Hard to say how much without emotional context, but it's worth being aware of.")
    
    # NLP-specific
    if mode == "formula_plus_nlp" and nlp:
        toxicity = nlp.get("toxicity_mean", 0)
        if toxicity > 0.3:
            risks.append("Some messages scored high on negativity. This could be playful sarcasm or actual friction — context matters.")
    
    # Low stability
    stability = scores.get("stability", {}).get("normalized", 0)
    if stability < 40:
        risks.append("Communication is inconsistent — long gaps or irregular patterns. That's fine if it works for you, but it can make connection harder to maintain.")
    
    # Fallback
    if len(risks) == 0:
        risks.append("No major red flags jump out from the data.")
    
    return risks[:5]  # Cap at 5


def _generate_suggestions(
    summary: Dict,
    scores: Dict,
    risks: List[str],
    mode: str
) -> List[str]:
    """Generate gentle nudges (not direct advice)."""
    suggestions = []
    
    # Based on risks
    if any("one-sided" in r.lower() or "uneven effort" in r.lower() for r in risks):
        suggestions.append("More balanced initiation could make the dynamic feel more mutual, if that's something you want.")
    
    if any("slow" in r.lower() or "response time" in r.lower() for r in risks):
        suggestions.append("If response times matter to you, it might be worth having a light conversation about communication preferences.")
    
    if any("warmth" in r.lower() or "emotional distance" in r.lower() for r in risks):
        suggestions.append("A bit more expressive communication (emojis, affirmations, enthusiasm) can go a long way if warmth is important to you.")
    
    if any("tension" in r.lower() or "conflict" in r.lower() or "negative" in r.lower() for r in risks):
        suggestions.append("When friction shows up, addressing it directly (but gently) tends to work better than letting it simmer.")
    
    if any("inconsistent" in r.lower() or "gaps" in r.lower() for r in risks):
        suggestions.append("If consistency matters, even small check-ins during quieter stretches can help maintain connection.")
    
    # Based on strengths
    engagement = scores.get("engagement", {}).get("normalized", 0)
    if engagement >= 70:
        suggestions.append("You're already keeping things active — maintaining that rhythm is probably more important than ramping up intensity.")
    
    # Fallback
    if len(suggestions) == 0:
        suggestions.append("The patterns here don't scream for major changes. If things feel good, keep doing what you're doing.")
    
    return suggestions[:4]  # Cap at 4


def _extract_sample_messages(nlp: Dict, max_count: int) -> List[Dict[str, str]]:
    """
    Extract sample messages for context (NLP mode only).
    Currently placeholder - would need message-level data from report.
    """
    # This would require storing sample messages in the report payload
    # For now, return empty since we don't have message-level data in report
    return []


if __name__ == "__main__":
    # Test with mock report
    mock_report = {
        "mode": "formula_only",
        "summary": {
            "total_messages": 500,
            "days_active": 30,
            "messages_per_day": 16.7,
            "dominant_sender": None,
            "relationship_type": "friends",
            "relationship_confidence": 0.8,
        },
        "scores": {
            "overall_health": {"normalized": 72},
            "engagement": {"normalized": 65},
            "warmth": {"normalized": 68},
            "conflict": {"normalized": 15},
            "stability": {"normalized": 58},
        },
        "type_prediction": {
            "type": "friends",
            "confidence": 0.8,
            "evidence": ["Moderate warmth", "High engagement"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {
                    "message_count": 250,
                    "initiation_count": 15,
                    "median_response_time_seconds": 300,
                    "emoji_stats": {"total": 45, "romantic": 5, "positive": 20}
                },
                "Bob": {
                    "message_count": 250,
                    "initiation_count": 18,
                    "median_response_time_seconds": 450,
                    "emoji_stats": {"total": 38, "romantic": 3, "positive": 15}
                },
            }
        },
        "nlp_metrics": None,
    }
    
    insights = generate_insights(mock_report, "formula_only")
    
    print("=== INSIGHTS TEST ===\n")
    print(f"Summary:\n{insights['summary']}\n")
    print(f"\nStrengths:")
    for s in insights['strengths']:
        print(f"  • {s}")
    print(f"\nRisks:")
    for r in insights['risks']:
        print(f"  • {r}")
    print(f"\nSuggestions:")
    for s in insights['suggestions']:
        print(f"  • {s}")
