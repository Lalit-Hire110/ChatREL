"""
ChatREL v4 - WhatsApp Chat Relationship Analyzer

A lightweight relationship analyzer using HuggingFace Inference API
for sentiment and toxicity detection, combined with local heuristics.

Designed for Hinglish and Romanized Marathi support with minimal dependencies.
"""

__version__ = "4.0.0"
__author__ = "ChatREL Team"

from . import config
from . import parser
from . import hf_client
from . import text_features
from . import message_processor
from . import aggregator
from . import scoring

__all__ = [
    "config",
    "parser",
    "hf_client",
    "text_features",
    "message_processor",
    "aggregator",
    "scoring",
]
