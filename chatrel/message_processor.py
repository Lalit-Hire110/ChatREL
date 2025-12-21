"""
Message processor for ChatREL v4
Combines HF API outputs with local heuristics (sensor fusion)
"""

import logging
from typing import List, Dict, Any
import pandas as pd

from . import config
from .hf_client import HFClient
from .text_features import extract_all_features

logger = logging.getLogger(__name__)


# Labels that indicate toxicity
TOXIC_LABELS = {"toxic", "hate", "abusive", "offensive"}


def extract_toxicity_score(prediction: Dict[str, Any]) -> float:
    """
    Convert HF toxicity prediction into a [0,1] toxicity score.
    
    HuggingFace toxicity models return predictions like:
    - {"label": "non-toxic", "score": 0.97} means 97% confident it's non-toxic
    - {"label": "toxic", "score": 0.8} means 80% confident it's toxic
    
    We need to convert this to a toxicity score where:
    - 0.0 = definitely non-toxic
    - 1.0 = definitely toxic
    
    Args:
        prediction: Dict with "label" and "score" keys from HF API
        
    Returns:
        Float in [0, 1] representing toxicity level
    """
    label = prediction.get("label", "").lower()
    score = prediction.get("score", 0.5)
    
    # IMPORTANT: Check non-toxic first before toxic (since "non-toxic" contains "toxic")
    # If label indicates non-toxicity, invert the score
    # e.g., "non-toxic" with score 0.97 → toxicity = 0.03
    if "non" in label or label == "clean" or label == "neutral":
        return 1.0 - score
    
    # If label indicates toxicity, use score directly
    if any(toxic_word in label for toxic_word in TOXIC_LABELS):
        return score
    
    # Fallback for unknown labels
    logger.warning(f"Unknown toxicity label: {label}, using score as-is")
    return score


class MessageProcessor:
    """Process messages with HF API + local heuristics."""
    
    def __init__(self, hf_client: HFClient):
        """
        Initialize processor.
        
        Args:
            hf_client: Initialized HF client
        """
        self.hf_client = hf_client
    
    def process_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all messages in DataFrame.
        
        Args:
            df: DataFrame with 'text' column
        
        Returns:
            DataFrame with added columns for sentiment, toxicity, features, and fusion results
        """
        # Graceful degradation: Skip NLP processing if disabled
        if not config.USE_NLP:
            logger.info("NLP disabled (USE_NLP=False), returning DataFrame with placeholder columns")
            df = df.copy()
            df['sentiment_label'] = 'neutral'
            df['sentiment_score'] = 0.5
            df['toxicity_label'] = 'non-toxic'
            df['toxicity_score'] = 0.0
            df['combined_sentiment'] = 0.0
            df['conflict_flag'] = False
            df['teasing_flag'] = False
            df['sentiment_uncertain'] = True
            df['emoji_count'] = 0
            df['emoji_valence'] = 0.0
            df['laughter_flag'] = False
            df['code_mixed'] = False
            df['has_slang'] = False
            df['has_romantic'] = False
            df['word_count'] = 0
            return df
        
        logger.info(f"Processing {len(df)} messages...")
        
        # Extract text (skip media/system messages)
        # Handle missing columns safely
        is_media = df["is_media"] if "is_media" in df.columns else pd.Series(False, index=df.index)
        is_system = df["is_system"] if "is_system" in df.columns else pd.Series(False, index=df.index)
        
        valid_mask = ~is_media & ~is_system & (df["text"].str.len() > 0)
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            logger.warning("No valid text messages to process")
            return df
        
        texts = valid_df["text"].tolist()
        
        # Deduplicate texts to save API calls
        unique_texts = list(set(texts))
        logger.info(f"Deduplicated messages: {len(texts)} -> {len(unique_texts)} unique texts")
        
        # Get HF predictions for unique texts
        logger.info("Querying sentiment model...")
        unique_sentiments = self.hf_client.get_sentiment(unique_texts)
        sent_map = {text: sent for text, sent in zip(unique_texts, unique_sentiments)}
        
        logger.info("Querying toxicity model...")
        unique_toxicities = self.hf_client.get_toxicity(unique_texts)
        tox_map = {text: tox for text, tox in zip(unique_texts, unique_toxicities)}
        
        # Map back to full list
        sentiments = [sent_map[text] for text in texts]
        toxicities = [tox_map[text] for text in texts]
        
        # Extract local features (fast enough to do for all, or could dedup too)
        logger.info("Extracting text features...")
        features_list = [extract_all_features(text) for text in texts]
        
        # Combine results
        logger.info("Applying fusion rules...")
        for i, (sent, tox, feats) in enumerate(zip(sentiments, toxicities, features_list)):
            idx = valid_df.index[i]
            
            # Store raw HF outputs
            df.loc[idx, "sentiment_label"] = sent["label"]
            df.loc[idx, "sentiment_score"] = sent["score"]
            df.loc[idx, "toxicity_label"] = tox["label"]
            # Use helper to correctly map toxicity score based on label
            df.loc[idx, "toxicity_score"] = extract_toxicity_score(tox)
            
            # Store features
            df.loc[idx, "emoji_count"] = feats["emoji_count"]
            df.loc[idx, "emoji_valence"] = feats["emoji_valence"]
            df.loc[idx, "laughter_flag"] = feats["laughter_flag"]
            df.loc[idx, "code_mixed"] = feats["is_code_mixed"]
            df.loc[idx, "has_slang"] = feats["has_slang"]
            df.loc[idx, "has_romantic"] = feats["has_romantic"]
            df.loc[idx, "word_count"] = feats["word_count"]
            
            # Apply fusion rules
            fusion_result = self._apply_fusion_rules(sent, tox, feats)
            
            df.loc[idx, "conflict_flag"] = fusion_result["conflict"]
            df.loc[idx, "teasing_flag"] = fusion_result["teasing"]
            df.loc[idx, "sentiment_uncertain"] = fusion_result["uncertain"]
            df.loc[idx, "combined_sentiment"] = fusion_result["combined_sentiment"]
        
        # Fill NaN for media/system messages
        df["sentiment_label"] = df["sentiment_label"].fillna("neutral")
        df["sentiment_score"] = df["sentiment_score"].fillna(0.5)
        df["toxicity_score"] = df["toxicity_score"].fillna(0.1)
        df["combined_sentiment"] = df["combined_sentiment"].fillna(0.5)
        
        logger.info("Message processing complete")
        return df
    
    def _apply_fusion_rules(
        self, 
        sentiment: Dict[str, Any], 
        toxicity: Dict[str, Any], 
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply sensor fusion rules.
        
        Returns dict with:
            - conflict: Boolean flag
            - teasing: Boolean flag
            - uncertain: Boolean flag
            - combined_sentiment: Final sentiment score [-1, 1]
        """
        sent_score = sentiment["score"]
        sent_label = sentiment["label"]
        # Use helper to get correct toxicity score
        tox_score = extract_toxicity_score(toxicity)
        
        # Convert sentiment label to numeric score
        if "positive" in sent_label.lower():
            sentiment_numeric = sent_score  # Already 0-1
        elif "negative" in sent_label.lower():
            sentiment_numeric = -sent_score  # Negate for negative
        else:
            sentiment_numeric = 0.0  # Neutral
        
        # Conflict detection
        conflict = tox_score > config.TOXICITY_CONFLICT_THRESHOLD
        
        # Uncertainty detection
        uncertain = sent_score < config.SENTIMENT_CONFIDENCE_THRESHOLD
        
        # Teasing detection (negative + laughter + low toxicity)
        is_negative = "negative" in sent_label.lower()
        teasing = (
            is_negative 
            and features["laughter_flag"] 
            and tox_score < config.TOXICITY_TEASING_THRESHOLD
        )
        
        # Combine sentiment with emoji valence
        emoji_val = features["emoji_valence"]
        
        # Weighted combination (70% model, 30% emoji)
        combined = 0.7 * sentiment_numeric + 0.3 * emoji_val
        
        # Slang boost (increase negative weight slightly if slang present)
        if features["has_slang"] and sentiment_numeric < 0:
            combined -= 0.1
        
        # Downweight if uncertain
        if uncertain:
            combined *= 0.5
        
        # Reduce conflict impact if teasing
        if teasing:
            conflict = False
        
        # Clip to [-1, 1]
        combined = max(-1.0, min(1.0, combined))
        
        return {
            "conflict": conflict,
            "teasing": teasing,
            "uncertain": uncertain,
            "combined_sentiment": combined,
        }


if __name__ == "__main__":
    # Test with mock client
    from .parser import WhatsAppParser
    
    client = HFClient(mock_mode=True)
    processor = MessageProcessor(client)
    
    # Test data
    test_text = """25/12/23, 09:15 - Alice: Good morning! ☀️ Hope you slept well
25/12/23, 09:18 - Bob: Morning yaar! ❤️ Slept amazing 😘
25/12/23, 09:20 - Alice: Haha you're funny 😂"""
    
    parser = WhatsAppParser()
    df = parser.parse_text(test_text)
    
    print("Before processing:")
    print(df[["sender", "text"]].head())
    
    df = processor.process_messages(df)
    
    print("\nAfter processing:")
    print(df[["sender", "text", "combined_sentiment", "conflict_flag"]].head())
