"""
Configuration module for ChatREL v4
Loads environment variables and provides default settings
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# HuggingFace API Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
TOXICITY_MODEL = os.getenv("TOXICITY_MODEL", "textdetox/xlmr-large-toxicity-classifier")

# API Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP", "0.4"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
HF_CONCURRENT_REQUESTS = int(os.getenv("HF_CONCURRENT_REQUESTS", "5"))

# Dev Server Settings
# Default to False on Windows if concurrency is high to prevent WinError 10038
_is_windows = os.name == 'nt'
_default_reload = "False" if (_is_windows and HF_CONCURRENT_REQUESTS > 1) else "True"
DEV_USE_RELOADER = os.getenv("CHATREL_DEV_RELOAD", _default_reload).lower() == "true"

# Cache Settings
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))
CACHE_DIR = PROJECT_ROOT / os.getenv("CACHE_DIR", ".cache")

# Privacy Settings
SEND_RAW_TEXT_TO_HF = os.getenv("SEND_RAW_TEXT_TO_HF", "True").lower() == "true"
PSEUDONYMIZE_NAMES = os.getenv("PSEUDONYMIZE_NAMES", "False").lower() == "true"

# Operational Mode Flags
USE_NLP = os.getenv("USE_NLP", "True").lower() == "true"
USE_INSIGHT_AI = os.getenv("USE_INSIGHT_AI", "False").lower() == "true"

# Demo Mode Configuration
DEMO_MODE = os.getenv("DEMO_MODE", "False").lower() == "true"
DEMO_CACHE_ENABLED = os.getenv("DEMO_CACHE_ENABLED", "True").lower() == "true"
DEMO_PROFILE = os.getenv("DEMO_PROFILE", "demo_v1")
DEMO_POLLUTE_CSM = os.getenv("DEMO_POLLUTE_CSM", "False").lower() == "true"
DEFAULT_RUN_MODE = os.getenv("DEFAULT_RUN_MODE", "normal")  # normal | csm | demo

# ============================================================================
# Contextual Sentiment Memory (CSM) Configuration
# ============================================================================

# Enable CSM caching and learning system
CSM_ENABLED = os.getenv("CSM_ENABLED", "True").lower() == "true"

# HuggingFace API Settings for CSM
LIVE_HF_ENABLED = os.getenv("LIVE_HF_ENABLED", "False").lower() == "true"

# Confidence threshold for inference (0.0 - 1.0)
# Below this, HF API is called (if LIVE_HF_ENABLED=True)
CSM_CONFIDENCE_THRESHOLD = float(os.getenv("CSM_CONFIDENCE_THRESHOLD", "0.70"))

# Variance-aware confidence adjustment
CSM_VAR_CONFIDENCE_WEIGHT = float(os.getenv("CSM_VAR_CONFIDENCE_WEIGHT", "1.0"))
CSM_VARIANCE_THRESHOLD = float(os.getenv("CSM_VARIANCE_THRESHOLD", "0.35"))

# Async learning settings
CSM_ASYNC_LEARNING = os.getenv("CSM_ASYNC_LEARNING", "True").lower() == "true"

# Privacy mode (store only hashes, not raw text)
CSM_PRIVACY_MODE = os.getenv("CSM_PRIVACY_MODE", "False").lower() == "true"

# Redis configuration
CSM_REDIS_URL = os.getenv("CSM_REDIS_URL", "redis://localhost:6379/1")
CSM_REDIS_TTL_HOURS = int(os.getenv("CSM_REDIS_TTL_HOURS", "24"))
CSM_REDIS_SYNC_INTERVAL_HOURS = int(os.getenv("CSM_REDIS_SYNC_INTERVAL_HOURS", "6"))

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", CSM_REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CSM_REDIS_URL)
CELERY_BROKER_CONNECTION_TIMEOUT = int(os.getenv("CELERY_BROKER_CONNECTION_TIMEOUT", "10"))
CELERY_BROKER_HEARTBEAT = int(os.getenv("CELERY_BROKER_HEARTBEAT", "30"))
BROKER_POOL_LIMIT = int(os.getenv("BROKER_POOL_LIMIT", "3"))
BROKER_CONNECTION_RETRY = os.getenv("BROKER_CONNECTION_RETRY", "True").lower() == "true"

# Database configuration
CSM_DB_PATH = os.getenv("CSM_DB_PATH", str(PROJECT_ROOT / "chatrel_csm.db"))

# Model versioning (bump to invalidate cache)
CSM_SENTIMENT_MODEL_VERSION = os.getenv("CSM_SENTIMENT_MODEL_VERSION", "1.0")
CSM_TOXICITY_MODEL_VERSION = os.getenv("CSM_TOXICITY_MODEL_VERSION", "1.0")

# Token statistics thresholds
CSM_MIN_TOKEN_COUNT = int(os.getenv("CSM_MIN_TOKEN_COUNT", "5"))
CSM_MIN_CONTEXT_COUNT = int(os.getenv("CSM_MIN_CONTEXT_COUNT", "3"))

# Rate limiting for HF API
MAX_HF_CALLS_PER_MIN = int(os.getenv("MAX_HF_CALLS_PER_MIN", "50"))
HF_THROTTLE_WINDOW_SECONDS = int(os.getenv("HF_THROTTLE_WINDOW_SECONDS", "60"))

# Decision logging (debug mode)
CSM_DEBUG_DECISIONS = os.getenv("CSM_DEBUG_DECISIONS", "False").lower() == "true"

# ============================================================================

# Scoring Weights (normalized to sum to 1.0)
WARMTH_WEIGHT = float(os.getenv("WARMTH_WEIGHT", "0.45"))
ENGAGEMENT_WEIGHT = float(os.getenv("ENGAGEMENT_WEIGHT", "0.30"))
CONFLICT_WEIGHT = float(os.getenv("CONFLICT_WEIGHT", "0.20"))
STABILITY_WEIGHT = float(os.getenv("STABILITY_WEIGHT", "0.05"))

# Windowing
DEFAULT_WINDOW_SIZE = int(os.getenv("DEFAULT_WINDOW_SIZE", "100"))
DEFAULT_TIME_WINDOW_DAYS = int(os.getenv("DEFAULT_TIME_WINDOW_DAYS", "7"))

# Emoji Lexicon (from ChatREL v3_1, weights scaled to [-1, 1])
EMOJI_LEXICON: Dict[str, float] = {
    # Romantic (high positive)
    "❤️": 1.0, "😍": 1.0, "💖": 1.0, "💕": 1.0, "💘": 1.0, "💓": 1.0,
    "💝": 1.0, "💞": 1.0, "😘": 1.0, "🥰": 1.0, "💗": 1.0, "💌": 1.0,
    "🤗": 0.8, "💟": 0.9, "💋": 0.9,
    
    # Affectionate (moderate positive)
    "😊": 0.6, "😇": 0.6, "💛": 0.7, "💙": 0.7, "💚": 0.7, "💜": 0.7,
    "🫶": 0.8, "🤝": 0.5, "🌹": 0.8, "🌸": 0.7, "💐": 0.8, "🫂": 0.8,
    
    # Playful/Laughter (moderate positive)
    "😂": 0.4, "🤣": 0.4, "😹": 0.4, "😝": 0.3, "😜": 0.3, "😛": 0.3,
    "🙃": 0.2, "😏": 0.2, "🤪": 0.3, "😺": 0.5, "😸": 0.5, "😻": 0.6,
    
    # Positive approval (low positive)
    "👍": 0.4, "👏": 0.5, "💯": 0.6, "✅": 0.5, "👌": 0.4, "🙏": 0.6,
    "🎉": 0.7, "✨": 0.6, "💫": 0.6, "🏆": 0.7, "🥳": 0.8, "💪": 0.5,
    
    # Excited/Intense (moderate positive)
    "😃": 0.5, "😄": 0.5, "😁": 0.5, "🤩": 0.8, "😎": 0.5, "🔥": 0.6,
    
    # Neutral
    "😐": 0.0, "😑": 0.0, "😶": 0.0, "🤔": 0.0, "🙄": -0.1, "😬": -0.1,
    "😕": -0.2, "😳": 0.0, "😌": 0.2, "😴": 0.0, "💀": 0.0, "☠️": -0.1,
    
    # Negative (moderate negative)
    "😡": -1.0, "🤬": -1.0, "😠": -1.0, "👿": -0.9, "💢": -0.8, "😤": -0.7,
    "😾": -0.7, "👎": -0.6, "☹️": -0.5, "😖": -0.5, "😈": -0.4,
    
    # Sad (moderate negative)
    "😢": -0.6, "😭": -0.7, "😞": -0.5, "🥺": -0.4, "😿": -0.6, "💔": -0.9,
    "😪": -0.3, "😥": -0.5, "😓": -0.4, "😰": -0.6, "😟": -0.5, "😔": -0.5,
}

# Romantic keywords (multilingual)
ROMANTIC_KEYWORDS = [
    "love", "miss", "jaan", "baby", "sweetheart", "pyaar", "dil", "ishq",
    "darling", "honey", "babe", "cuddle", "kiss", "hug", "soulmate",
]

# Hinglish tokens (for code-mixing detection)
HINGLISH_TOKENS = [
    "yaar", "kya", "bhai", "dost", "accha", "theek", "bas", "aur",
    "hai", "kaise", "kahan", "kyun", "matlab", "sahi", "nahi", "haan",
    "sala", "arre", "dekh", "sun", "chal", "chalo", "aaja", "aao",
]

# Marathi romanized tokens
MARATHI_TOKENS = [
    "ahe", "kay", "zhala", "bara", "mhanun", "kasa", "kuthe", "ata",
    "mag", "nahi", "ho", "na", "tula", "mala", "aplya", "tyala",
]

# Slang tokens (may indicate informal/negative tone)
SLANG_TOKENS = [
    "wtf", "damn", "shit", "fuck", "hell", "dude", "lol", "lmao",
    "bruh", "bro", "oof", "meh", "nope", "yep", "yeah", "nah",
]

# Laughter patterns
LAUGHTER_PATTERNS = ["lol", "lmao", "haha", "hehe", "hihi", "hoho", "😂", "🤣"]

# Sentiment thresholds
SENTIMENT_CONFIDENCE_THRESHOLD = 0.45  # Below this, mark as uncertain
TOXICITY_CONFLICT_THRESHOLD = 0.7      # Above this, flag as conflict
TOXICITY_TEASING_THRESHOLD = 0.2       # Below this, negative + laughter = teasing

# Relationship type thresholds
RELATIONSHIP_THRESHOLDS = {
    "couple": {
        "warmth_min": 0.7,
        "emoji_affinity_min": 0.6,
        "reciprocity_min": 0.5,
    },
    "crush": {
        "warmth_min": 0.65,
        "emoji_affinity_min": 0.5,
        "reciprocity_max": 0.7,  # Asymmetric
    },
    "friend": {
        "warmth_min": 0.4,
        "warmth_max": 0.65,
        "engagement_min": 0.5,
    },
    "family": {
        "warmth_min": 0.3,
        "warmth_max": 0.6,
        "avg_words_min": 10,
        "emoji_density_max": 0.5,
    },
}


def get_config_summary() -> Dict[str, Any]:
    """Return a summary of current configuration."""
    return {
        "models": {
            "sentiment": SENTIMENT_MODEL,
            "toxicity": TOXICITY_MODEL,
        },
        "api": {
            "batch_size": BATCH_SIZE,
            "rate_limit_sleep": RATE_LIMIT_SLEEP,
            "timeout": API_TIMEOUT,
            "max_retries": MAX_RETRIES,
            "concurrent_requests": HF_CONCURRENT_REQUESTS,
        },
        "dev_server": {
            "use_reloader": DEV_USE_RELOADER,
        },
        "cache": {
            "ttl_days": CACHE_TTL_DAYS,
            "dir": str(CACHE_DIR),
        },
        "privacy": {
            "send_raw_text": SEND_RAW_TEXT_TO_HF,
            "pseudonymize": PSEUDONYMIZE_NAMES,
        },
        "mode": {
            "use_nlp": USE_NLP,
            "use_insight_ai": USE_INSIGHT_AI,
            "demo_mode": DEMO_MODE,
            "default_run_mode": DEFAULT_RUN_MODE,
        },
        "scoring": {
            "warmth_weight": WARMTH_WEIGHT,
            "engagement_weight": ENGAGEMENT_WEIGHT,
            "conflict_weight": CONFLICT_WEIGHT,
            "stability_weight": STABILITY_WEIGHT,
        },
        "windowing": {
            "default_size": DEFAULT_WINDOW_SIZE,
            "default_days": DEFAULT_TIME_WINDOW_DAYS,
        },
    }


def validate_config() -> tuple[bool, str]:
    """Validate configuration. Returns (is_valid, message)."""
    if not HF_TOKEN and USE_NLP:
        return False, "HF_TOKEN not set in .env file (required when USE_NLP=True)"
    
    total_weight = WARMTH_WEIGHT + ENGAGEMENT_WEIGHT + CONFLICT_WEIGHT + STABILITY_WEIGHT
    if abs(total_weight - 1.0) > 0.01:
        return False, f"Scoring weights sum to {total_weight:.2f}, should be ~1.0"
    
    return True, "Configuration valid"


if __name__ == "__main__":
    # Print config summary for debugging
    import json
    print("ChatREL v4 Configuration:")
    print(json.dumps(get_config_summary(), indent=2))
    print()
    valid, msg = validate_config()
    print(f"Validation: {msg}")
