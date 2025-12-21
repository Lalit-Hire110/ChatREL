"""
ChatREL Query Engine
Rule-based Q&A system for relationship reports (no LLM)
"""

import re
from typing import Dict, Any, List, Tuple, Optional


def answer_query(report: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Answer a user question about their relationship report.
    
    Uses only rule-based logic and the report dict - no LLM.
    
    Args:
        report: Full report from generate_report()
        question: User's question
        
    Returns:
        {
            "answer": str,
            "intent": str,
            "target": str or None,
            "provenance": [{"type": "metric", "key": "...", "value": ...}, ...],
            "confidence": float (0-1),
            "error": str or None
        }
    """
    # Classify intent
    intent, target = classify_intent(question)
    
    # Generate answer
    result = generate_answer(report, intent, target, question)
    
    return result


def classify_intent(question: str) -> Tuple[str, Optional[str]]:
    """
    Classify question intent using keyword matching.
    
    Returns:
        (intent_name, target_dimension or None)
        
    Intents:
        - EXPLAIN_SCORE
        - WHY_LOW_SCORE / WHY_HIGH_SCORE
        - WHO_TEXTS_MORE
        - TIMING_PATTERN
        - HOW_TO_IMPROVE
        - REL_LABEL_WHY
        - HELP_CAPABILITIES
        - OUT_OF_SCOPE
    """
    q = question.lower().strip()
    
    # Extract target dimension if mentioned
    target = None
    if "engagement" in q:
        target = "engagement"
    elif "warmth" in q:
        target = "warmth"
    elif "conflict" in q:
        target = "conflict"
    elif "stability" in q or "consistent" in q:
        target = "stability"
    elif "health" in q or "overall" in q:
        target = "overall_health"
    
    # Out of scope (safety first) - check for patterns AND keywords
    out_of_scope_keywords = [
        "break up", "breakup", "leave", "divorce", "love me", "loves me",
        "cheat", "marry", "propose"
    ]
    out_of_scope_patterns = [
        r"will (this|it|the relationship|our relationship|we|[a-z]* relationship) last",
        r"should i (leave|stay|end|break)",
        r"(does|do) (she|he|they) love me",
        r"future.*relationship"
    ]
    
    if any(keyword in q for keyword in out_of_scope_keywords):
        return ("OUT_OF_SCOPE", None)
    if any(re.search(pattern, q) for pattern in out_of_scope_patterns):
        return ("OUT_OF_SCOPE", None)
    
    # Help/capabilities - make more specific to avoid false matches
    if re.search(r"(what can (you|i ask)|^help$|capabilities|what.*(can you|should i ask))", q):
        return ("HELP_CAPABILITIES", None)
    
    # Why relationship label
    if re.search(r"why.*(couple|friend|family|one-sided|acquaintance)", q) or \
       re.search(r"why.*(call|say|label|classify)", q):
        return ("REL_LABEL_WHY", None)
    
    # Why low/high score
    if re.search(r"why.*(low|bad|poor|weak)", q):
        return ("WHY_LOW_SCORE", target)
    if re.search(r"why.*(high|good|strong)", q):
        return ("WHY_HIGH_SCORE", target)
    
    # Explain score
    if re.search(r"what (is|does|means?)|explain|define", q) and target:
        return ("EXPLAIN_SCORE", target)
    
    # Who texts more
    if re.search(r"who (text|message|talk|write)s? more|imbalanc|one.?sided|effort|put.*more", q):
        return ("WHO_TEXTS_MORE", None)
    
    # Timing patterns
    if re.search(r"(who|response|reply|respond).*(fast|slow|quick|long)|take.*to reply", q):
        return ("TIMING_PATTERN", None)
    
    # How to improve
    if re.search(r"how (can|do|to).*(improve|fix|better|increase)", q):
        return ("HOW_TO_IMPROVE", target)
    
    # Default: try to explain if target found, otherwise help
    if target:
        return ("EXPLAIN_SCORE", target)
    else:
        return ("HELP_CAPABILITIES", None)


def generate_answer(
    report: Dict[str, Any],
    intent: str,
    target: Optional[str],
    question: str
) -> Dict[str, Any]:
    """Generate answer based on intent and report data."""
    
    if intent == "EXPLAIN_SCORE":
        return _explain_score(report, target)
    elif intent == "WHY_LOW_SCORE":
        return _why_low_score(report, target)
    elif intent == "WHY_HIGH_SCORE":
        return _why_high_score(report, target)
    elif intent == "WHO_TEXTS_MORE":
        return _who_texts_more(report)
    elif intent == "TIMING_PATTERN":
        return _timing_pattern(report)
    elif intent == "HOW_TO_IMPROVE":
        return _how_to_improve(report, target)
    elif intent == "REL_LABEL_WHY":
        return _rel_label_why(report)
    elif intent == "HELP_CAPABILITIES":
        return _help_capabilities()
    elif intent == "OUT_OF_SCOPE":
        return _out_of_scope()
    else:
        return _help_capabilities()


def _explain_score(report: Dict[str, Any], target: Optional[str]) -> Dict[str, Any]:
    """Explain what a score dimension means."""
    if not target:
        return {
            "answer": "I can explain engagement, warmth, conflict, stability, or overall health. Try asking 'What is engagement?'",
            "intent": "EXPLAIN_SCORE",
            "target": None,
            "provenance": [],
            "confidence": 0.5,
            "error": None
        }
    
    scores = report.get("scores", {})
    score_data = scores.get(target, {})
    value = score_data.get("normalized", 0)
    
    explanations = {
        "engagement": "Engagement measures how active and balanced the chat is: how often you talk, how evenly you both contribute, and how quickly you reply.",
        "warmth": "Warmth measures emotional positivity and affection: things like positive emojis, kind language, and supportive tone.",
        "conflict": "Conflict measures negativity and tension: negative emojis, harsh language, or tense exchanges. Lower is better here.",
        "stability": "Stability measures consistency over time: how regularly you chat, whether you've maintained contact, and if there are long gaps.",
        "overall_health": "Overall health is a weighted combination of engagement, warmth, conflict (inverted), and stability. It's an overall measure of how the relationship is going based on communication patterns."
    }
    
    explanation = explanations.get(target, "I'm not sure about that dimension.")
    
    # Add current value
    if value <= 30:
        band = "on the lower side"
    elif value <= 60:
        band = "moderate"
    else:
        band = "on the higher side"
    
    if target == "conflict":
        # Invert for conflict
        if value <= 20:
            band = "low, which is good"
        elif value <= 50:
            band = "moderate"
        else:
            band = "high"
    
    answer = f"{explanation} Your {target} score is {int(value)}, which is {band}."
    
    provenance = [
        {"type": "metric", "key": f"scores.{target}.normalized", "value": int(value)}
    ]
    
    return {
        "answer": answer,
        "intent": "EXPLAIN_SCORE",
        "target": target,
        "provenance": provenance,
        "confidence": 0.95,
        "error": None
    }


def _why_low_score(report: Dict[str, Any], target: Optional[str]) -> Dict[str, Any]:
    """Explain why a score is low."""
    if not target:
        return {
            "answer": "Which score are you asking about? Try 'Why is my engagement low?' or 'Why is stability low?'",
            "intent": "WHY_LOW_SCORE",
            "target": None,
            "provenance": [],
            "confidence": 0.3,
            "error": "Please specify which score"
        }
    
    scores = report.get("scores", {})
    score_data = scores.get(target, {})
    value = score_data.get("normalized", 0)
    inputs = score_data.get("inputs", {})
    
    provenance = [
        {"type": "metric", "key": f"scores.{target}.normalized", "value": int(value)}
    ]
    
    # Build explanation based on target
    reasons = []
    
    if target == "engagement":
        msgs_per_day = inputs.get("msgs_per_day", 0)
        balance = inputs.get("balance_score", 100)
        
        if msgs_per_day < 5:
            reasons.append(f"you only chat about {msgs_per_day:.1f} times per day")
            provenance.append({"type": "metric", "key": "msgs_per_day", "value": round(msgs_per_day, 1)})
        if balance < 70:
            reasons.append("the message balance is uneven")
            provenance.append({"type": "metric", "key": "balance_score", "value": int(balance)})
    
    elif target == "warmth":
        emoji_aff = inputs.get("emoji_affection_ratio", 0)
        avg_words = inputs.get("avg_words_per_message", 0)
        
        if emoji_aff < 0.3:
            reasons.append("relatively few positive or romantic emojis")
            provenance.append({"type": "metric", "key": "emoji_affection_ratio", "value": round(emoji_aff, 2)})
        if avg_words < 10:
            reasons.append("messages tend to be short")
            provenance.append({"type": "metric", "key": "avg_words_per_message", "value": round(avg_words, 1)})
    
    elif target == "stability":
        days = report["summary"].get("days_active", 0)
        msgs = report["summary"].get("total_messages", 0)
        
        if days < 14:
            reasons.append(f"the chat is only {days} days old")
            provenance.append({"type": "metric", "key": "days_active", "value": days})
        if msgs < 100:
            reasons.append(f"only {msgs} messages total")
            provenance.append({"type": "metric", "key": "total_messages", "value": msgs})
    
    if not reasons:
        reasons.append("several factors combined")
    
    answer = f"Your {target} score is {int(value)}, which is low. This is mainly because {', and '.join(reasons)}."
    
    return {
        "answer": answer,
        "intent": "WHY_LOW_SCORE",
        "target": target,
        "provenance": provenance,
        "confidence": 0.85,
        "error": None
    }


def _why_high_score(report: Dict[str, Any], target: Optional[str]) -> Dict[str, Any]:
    """Explain why a score is high."""
    if not target:
        return {
            "answer": "Which score? Try asking about a specific dimension like engagement or warmth.",
            "intent": "WHY_HIGH_SCORE",
            "target": None,
            "provenance": [],
            "confidence": 0.3,
            "error": "Please specify which score"
        }
    
    scores = report.get("scores", {})
    score_data = scores.get(target, {})
    value = score_data.get("normalized", 0)
    inputs = score_data.get("inputs", {})
    
    provenance = [
        {"type": "metric", "key": f"scores.{target}.normalized", "value": int(value)}
    ]
    
    reasons = []
    
    if target == "engagement":
        msgs_per_day = inputs.get("msgs_per_day", 0)
        if msgs_per_day > 10:
            reasons.append(f"you chat frequently ({msgs_per_day:.1f} messages/day)")
            provenance.append({"type": "metric", "key": "msgs_per_day", "value": round(msgs_per_day, 1)})
    
    elif target == "warmth":
        emoji_aff = inputs.get("emoji_affection_ratio", 0)
        if emoji_aff > 0.5:
            reasons.append("lots of positive and romantic emojis")
            provenance.append({"type": "metric", "key": "emoji_affection_ratio", "value": round(emoji_aff, 2)})
    
    if not reasons:
        reasons.append("strong performance across multiple factors")
    
    answer = f"Your {target} score is {int(value)}, which is high. This is because {', and '.join(reasons)}."
    
    return {
        "answer": answer,
        "intent": "WHY_HIGH_SCORE",
        "target": target,
        "provenance": provenance,
        "confidence": 0.85,
        "error": None
    }


def _who_texts_more(report: Dict[str, Any]) -> Dict[str, Any]:
    """Answer who texts more."""
    structural = report.get("structural_metrics", {})
    per_sender = structural.get("per_sender", {})
    
    if len(per_sender) < 2:
        return {
            "answer": "I need at least two people in the chat to compare.",
            "intent": "WHO_TEXTS_MORE",
            "target": None,
            "provenance": [],
            "confidence": 0.2,
            "error": "Not enough senders"
        }
    
    senders = list(per_sender.keys())
    counts = [per_sender[s]["message_count"] for s in senders]
    total = sum(counts)
    
    if counts[0] > counts[1]:
        dominant = senders[0]
        ratio = counts[0] / total
    else:
        dominant = senders[1]
        ratio = counts[1] / total
    
    provenance = [
        {"type": "metric", "key": f"messages.{senders[0]}", "value": counts[0]},
        {"type": "metric", "key": f"messages.{senders[1]}", "value": counts[1]},
    ]
    
    if ratio > 0.75:
        answer = f"{dominant} sends about {ratio*100:.0f}% of the messages. That's a pretty significant imbalance — they're carrying most of the conversation."
    elif ratio > 0.6:
        answer = f"{dominant} sends about {ratio*100:.0f}% of the messages, so they contribute more, but it's not extremely one-sided."
    else:
        answer = f"It's pretty balanced — {senders[0]} sends {counts[0]} messages and {senders[1]} sends {counts[1]}. Neither person dominates."
    
    return {
        "answer": answer,
        "intent": "WHO_TEXTS_MORE",
        "target": None,
        "provenance": provenance,
        "confidence": 0.95,
        "error": None
    }


def _timing_pattern(report: Dict[str, Any]) -> Dict[str, Any]:
    """Answer about response times."""
    structural = report.get("structural_metrics", {})
    per_sender = structural.get("per_sender", {})
    
    if len(per_sender) < 2:
        return {
            "answer": "I need at least two people to compare response times.",
            "intent": "TIMING_PATTERN",
            "target": None,
            "provenance": [],
            "confidence": 0.2,
            "error": "Not enough senders"
        }
    
    senders = list(per_sender.keys())
    times_sec = [per_sender[s]["median_response_time_seconds"] for s in senders]
    
    def format_time(seconds):
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds/60)} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    answer = f"On average, {senders[0]} replies in about {format_time(times_sec[0])}, while {senders[1]} replies in about {format_time(times_sec[1])}. "
    
    if times_sec[0] < times_sec[1] * 0.5:
        answer += f"That suggests {senders[0]} tends to respond much faster."
    elif times_sec[1] < times_sec[0] * 0.5:
        answer += f"That suggests {senders[1]} tends to respond much faster."
    else:
        answer += "Response times are fairly similar between you two."
    
    provenance = [
        {"type": "metric", "key": f"response_time.{senders[0]}", "value": int(times_sec[0])},
        {"type": "metric", "key": f"response_time.{senders[1]}", "value": int(times_sec[1])},
    ]
    
    return {
        "answer": answer,
        "intent": "TIMING_PATTERN",
        "target": None,
        "provenance": provenance,
        "confidence": 0.9,
        "error": None
    }


def _how_to_improve(report: Dict[str, Any], target: Optional[str]) -> Dict[str, Any]:
    """Suggest how to improve (gentle nudges)."""
    scores = report.get("scores", {})
    
    suggestions = []
    
    if target:
        score = scores.get(target, {}).get("normalized", 50)
        
        if target == "engagement" and score < 60:
            suggestions.append("You might find more frequent, shorter check-ins helpful for keeping the chat active.")
        elif target == "warmth" and score < 60:
            suggestions.append("It could be worth expressing appreciation more clearly or using emojis to add warmth.")
        elif target == "stability" and score < 60:
            suggestions.append("Consistency tends to matter more than intensity — even small regular check-ins can help.")
        elif target == "conflict" and score > 40:
            suggestions.append("When friction shows up, addressing it gently and directly tends to work better than letting it sit.")
    
    if not suggestions:
        overall = scores.get("overall_health", {}).get("normalized", 50)
        if overall >= 70:
            suggestions.append("Things look pretty good overall. The main thing is keeping up what you're already doing.")
        else:
            suggestions.append("Focus on consistency, clearer communication, and mutual effort. Small improvements tend to compound over time.")
    
    answer = " ".join(suggestions)
    
    return {
        "answer": answer,
        "intent": "HOW_TO_IMPROVE",
        "target": target,
        "provenance": [],
        "confidence": 0.7,
        "error": None
    }


def _rel_label_why(report: Dict[str, Any]) -> Dict[str, Any]:
    """Explain why a relationship label was chosen."""
    type_pred = report.get("type_prediction", {})
    rel_type = type_pred.get("type", "unknown")
    confidence = type_pred.get("confidence", 0)
    evidence = type_pred.get("evidence", [])
    
    answer = f"I labeled this as '{rel_type}' with {confidence*100:.0f}% confidence. "
    
    if evidence:
        answer += "The main reasons: " + "; ".join(evidence[:3]) + "."
    else:
        answer += "This is based on the overall pattern of scores and metrics."
    
    return {
        "answer": answer,
        "intent": "REL_LABEL_WHY",
        "target": None,
        "provenance": [
            {"type": "prediction", "key": "relationship_type", "value": rel_type},
            {"type": "prediction", "key": "confidence", "value": round(confidence, 2)}
        ],
        "confidence": 0.9,
        "error": None
    }


def _help_capabilities() -> Dict[str, Any]:
    """Explain what the assistant can do."""
    answer = ("I can help explain your scores (engagement, warmth, conflict, stability), "
              "show who texts more, compare response times, and explain why a certain "
              "relationship label was chosen. Try asking things like 'Why is my conflict score high?' "
              "or 'Who texts more?' or 'What does engagement mean?'")
    
    return {
        "answer": answer,
        "intent": "HELP_CAPABILITIES",
        "target": None,
        "provenance": [],
        "confidence": 1.0,
        "error": None
    }


def _out_of_scope() -> Dict[str, Any]:
    """Handle out-of-scope questions."""
    answer = ("I can't tell you whether to break up, whether someone loves you, or predict "
              "the future of the relationship. I can only explain patterns in the chat and "
              "how you both tend to communicate. If you'd like, ask me about scores, balance, "
              "or response times instead.")
    
    return {
        "answer": answer,
        "intent": "OUT_OF_SCOPE",
        "target": None,
        "provenance": [],
        "confidence": 1.0,
        "error": "Question is out of scope"
    }


if __name__ == "__main__":
    # Test with mock report
    mock_report = {
        "summary": {
            "total_messages": 150,
            "days_active": 20,
            "messages_per_day": 7.5,
            "dominant_sender": "Alice",
        },
        "scores": {
            "engagement": {"normalized": 55, "inputs": {"msgs_per_day": 7.5, "balance_score": 65}},
            "warmth": {"normalized": 68, "inputs": {"emoji_affection_ratio": 0.4}},
            "conflict": {"normalized": 15, "inputs": {}},
            "stability": {"normalized": 42, "inputs": {}},
            "overall_health": {"normalized": 58},
        },
        "type_prediction": {
            "type": "friends",
            "confidence": 0.75,
            "evidence": ["Moderate warmth", "Balanced messaging"]
        },
        "structural_metrics": {
            "per_sender": {
                "Alice": {"message_count": 95, "median_response_time_seconds": 300},
                "Bob": {"message_count": 55, "median_response_time_seconds": 1800},
            }
        }
    }
    
    tests = [
        "What is engagement?",
        "Why is my stability low?",
        "Who texts more?",
        "Who replies faster?",
        "Should I break up?",
    ]
    
    print("=== QUERY ENGINE TEST ===\n")
    for q in tests:
        result = answer_query(mock_report, q)
        print(f"Q: {q}")
        print(f"Intent: {result['intent']}, Target: {result['target']}")
        print(f"A: {result['answer']}\n")
