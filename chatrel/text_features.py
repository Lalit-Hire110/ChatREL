"""
Text feature extraction for ChatREL v4
Emoji parsing, laughter detection, code-mixing flags, and heuristics
"""

import re
import logging
from typing import Dict, Any, List
import emoji

from . import config

logger = logging.getLogger(__name__)


def extract_emojis(text: str) -> List[str]:
    """Extract all emojis from text."""
    # Use emoji library to extract
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    return emojis


def compute_emoji_features(text: str) -> Dict[str, Any]:
    """
    Compute emoji-based features.
    
    Returns:
        - emoji_count: Total emoji count
        - emoji_valence: Average sentiment score of emojis
        - emoji_list: List of emojis found
    """
    emojis = extract_emojis(text)
    
    if not emojis:
        return {
            "emoji_count": 0,
            "emoji_valence": 0.0,
            "emoji_list": [],
        }
    
    # Look up scores from lexicon
    scores = [config.EMOJI_LEXICON.get(e, 0.0) for e in emojis]
    avg_valence = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "emoji_count": len(emojis),
        "emoji_valence": avg_valence,
        "emoji_list": emojis,
    }


def detect_laughter(text: str) -> bool:
    """Detect laughter in text (emoji or text patterns)."""
    lower = text.lower()
    
    # Check text patterns
    for pattern in config.LAUGHTER_PATTERNS:
        if pattern.lower() in lower:
            return True
    
    return False


def detect_code_mixing(text: str) -> Dict[str, Any]:
    """
    Detect Hinglish/Marathi code-mixing based on romanized tokens.
    
    Returns:
        - is_code_mixed: Boolean flag
        - hinglish_tokens: List of detected Hinglish tokens
        - marathi_tokens: List of detected Marathi tokens
    """
    words = text.lower().split()
    
    hinglish_found = [w for w in words if w in [t.lower() for t in config.HINGLISH_TOKENS]]
    marathi_found = [w for w in words if w in [t.lower() for t in config.MARATHI_TOKENS]]
    
    is_mixed = len(hinglish_found) > 0 or len(marathi_found) > 0
    
    return {
        "is_code_mixed": is_mixed,
        "hinglish_tokens": hinglish_found,
        "marathi_tokens": marathi_found,
    }


def detect_slang(text: str) -> Dict[str, Any]:
    """
    Detect slang tokens.
    
    Returns:
        - has_slang: Boolean flag
        - slang_tokens: List of detected slang tokens
    """
    words = text.lower().split()
    slang_found = [w for w in words if w in [t.lower() for t in config.SLANG_TOKENS]]
    
    return {
        "has_slang": len(slang_found) > 0,
        "slang_tokens": slang_found,
    }


def detect_romantic_keywords(text: str) -> Dict[str, Any]:
    """
    Detect romantic keywords.
    
    Returns:
        - has_romantic: Boolean flag
        - romantic_keywords: List of detected keywords
    """
    words = text.lower().split()
    romantic_found = [w for w in words if w in [k.lower() for k in config.ROMANTIC_KEYWORDS]]
    
    return {
        "has_romantic": len(romantic_found) > 0,
        "romantic_keywords": romantic_found,
    }


def compute_text_flags(text: str) -> Dict[str, Any]:
    """
    Compute various text flags.
    
    Returns dict with:
        - question_flag: Has question mark
        - exclaim_flag: Has exclamation mark
        - all_caps_flag: Is majority uppercase
        - msg_length: Character count
        - word_count: Word count
    """
    if not text:
        return {
            "question_flag": False,
            "exclaim_flag": False,
            "all_caps_flag": False,
            "msg_length": 0,
            "word_count": 0,
        }
    
    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]
    
    return {
        "question_flag": "?" in text,
        "exclaim_flag": "!" in text,
        "all_caps_flag": len(alpha_chars) > 0 and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7,
        "msg_length": len(text),
        "word_count": len(words),
    }


def extract_all_features(text: str) -> Dict[str, Any]:
    """
    Extract all text features in one call.
    
    Returns combined dict of all features.
    """
    features = {}
    
    # Emoji features
    features.update(compute_emoji_features(text))
    
    # Laughter
    features["laughter_flag"] = detect_laughter(text)
    
    # Code mixing
    features.update(detect_code_mixing(text))
    
    # Slang
    features.update(detect_slang(text))
    
    # Romantic keywords
    features.update(detect_romantic_keywords(text))
    
    # Text flags
    features.update(compute_text_flags(text))
    
    return features


if __name__ == "__main__":
    # Test feature extraction
    test_texts = [
        "I love you yaar ❤️😘",
        "Aaj bara busy ahe, later baat karte hain",
        "WTF is going on here?! 😡",
        "Hahaha that's so funny 😂😂😂",
        "Meeting at 3 PM tomorrow",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        features = extract_all_features(text)
        for k, v in features.items():
            if v and k not in ["emoji_list", "hinglish_tokens", "marathi_tokens", "slang_tokens", "romantic_keywords"]:
                print(f"  {k}: {v}")
