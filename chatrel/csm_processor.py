"""
CSM-Enhanced Message Processor for ChatREL v4
Integrates Contextual Sentiment Memory into the message processing pipeline
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
import time

from .hf_client import HFClient
from .utils.nlp_cache import NLPCache
from .utils.token_stats import TokenStatsEngine
from .utils.decision_logger import DecisionLogger, Timer
from .tasks import enqueue_token_update
from . import config

logger = logging.getLogger(__name__)


class CSMMessageProcessor:
    """
    Message processor with CSM integration.
    
    Handles:
    - Cache lookup
    - Inference fallback
    - HF API calls (when needed)
    - Token statistics updates
    - Decision logging
    """
    
    def __init__(
        self,
        hf_client: Optional[HFClient] = None,
        use_csm: Optional[bool] = None,
        suppress_learning: bool = False
    ):
        """
        Initialize CSM message processor.
        
        Args:
            hf_client: HuggingFace client (optional, created if needed)
            use_csm: Enable CSM features (default from config)
            suppress_learning: If True, do not update token stats (e.g. for demo runs)
        """
        self.use_csm = use_csm if use_csm is not None else config.CSM_ENABLED
        self.hf_client = hf_client
        self.suppress_learning = suppress_learning
        
        # Initialize CSM components
        if self.use_csm:
            try:
                self.nlp_cache = NLPCache()
                self.token_engine = TokenStatsEngine()
                self.decision_logger = DecisionLogger()
                logger.info("CSM enabled - using cache + inference")
            except Exception as e:
                logger.warning(f"CSM initialization failed: {e} - falling back to standard mode")
                self.use_csm = False
        
        if not self.use_csm or (config.LIVE_HF_ENABLED and not self.hf_client):
            # Fallback to standard HF client
            if not self.hf_client:
                self.hf_client = HFClient(mock_mode=False, use_cache=True)
        
        logger.info(
            f"CSMMessageProcessor initialized (csm={'yes' if self.use_csm else 'no'}, "
            f"hf={'yes' if self.hf_client else 'no'})"
        )
    
    def process_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process messages with CSM integration.
        
        Args:
            df: DataFrame with 'text' column
        
        Returns:
            DataFrame with 'sentiment' and 'toxicity' columns added
        """
        df_result = df.copy()
        
        if not self.use_csm:
            # Fallback to standard processing
            return self._process_standard(df_result)
        
        # Process with CSM
        texts = df_result['text'].tolist()
        
        sentiment_scores = []
        toxicity_scores = []
        
        # New lists for compatibility
        sentiment_labels = []
        toxicity_labels = []
        combined_sentiments = []
        conflict_flags = []
        teasing_flags = []
        sentiment_uncertains = []
        
        # Feature columns
        emoji_counts = []
        emoji_valences = []
        laughter_flags = []
        code_mixeds = []
        has_slangs = []
        has_romantics = []
        word_counts = []
        
        cache_hits = 0
        inference_hits = 0
        hf_calls = 0
        
        logger.info(f"Processing {len(texts)} messages with CSM...")
        
        from .text_features import extract_all_features
        from .message_processor import extract_toxicity_score
        
        # Handle missing columns safely for filtering
        is_media = df_result["is_media"] if "is_media" in df_result.columns else pd.Series(False, index=df_result.index)
        is_system = df_result["is_system"] if "is_system" in df_result.columns else pd.Series(False, index=df_result.index)
        valid_mask = (~is_media & ~is_system & (df_result["text"].str.len() > 0)).tolist()

        for i, text in enumerate(texts):
            # Skip invalid messages (media, system, empty)
            if not valid_mask[i]:
                sentiment_scores.append(0.0)
                sentiment_labels.append("neutral")
                toxicity_scores.append(0.0)
                toxicity_labels.append("non-toxic")
                combined_sentiments.append(0.0)
                conflict_flags.append(False)
                teasing_flags.append(False)
                sentiment_uncertains.append(True)
                
                emoji_counts.append(0)
                emoji_valences.append(0.0)
                laughter_flags.append(False)
                code_mixeds.append(False)
                has_slangs.append(False)
                has_romantics.append(False)
                word_counts.append(0)
                continue

            with Timer() as total_timer:
                sentiment, toxicity, source_info = self._process_single_message(text)
                
                # Extract features
                feats = extract_all_features(text)
                
                # Prepare inputs for fusion
                # Construct pseudo-dicts to match MessageProcessor expectation
                sent_dict = {
                    "label": "positive" if sentiment > 0.05 else ("negative" if sentiment < -0.05 else "neutral"),
                    "score": abs(sentiment)
                }
                
                # Toxicity is already a score 0-1 from CSM pipeline
                # But fusion expects a dict with label/score to call extract_toxicity_score again?
                # No, we can adapt fusion or just pass a constructed dict that yields the score we have.
                # If we pass label="toxic", score=toxicity, extract_toxicity_score returns toxicity.
                tox_dict = {
                    "label": "toxic",
                    "score": toxicity
                }
                
                # Apply fusion
                fusion = self._apply_fusion_rules(sent_dict, tox_dict, feats)
                
                # Store results
                sentiment_scores.append(abs(sentiment)) # MessageProcessor stores absolute score usually? No, it stores confidence.
                # Wait, MessageProcessor stores sent["score"] which is confidence (0.5-1.0).
                # CSM returns signed score (-1 to 1).
                # We need to convert back if we want exact compatibility?
                # Actually, Aggregator uses combined_sentiment.
                # But it also uses sentiment_label and sentiment_score.
                
                sentiment_labels.append(sent_dict["label"])
                toxicity_scores.append(toxicity)
                toxicity_labels.append("toxic" if toxicity > 0.5 else "non-toxic")
                
                combined_sentiments.append(fusion["combined_sentiment"])
                conflict_flags.append(fusion["conflict"])
                teasing_flags.append(fusion["teasing"])
                sentiment_uncertains.append(fusion["uncertain"])
                
                # Features
                emoji_counts.append(feats["emoji_count"])
                emoji_valences.append(feats["emoji_valence"])
                laughter_flags.append(feats["laughter_flag"])
                code_mixeds.append(feats["is_code_mixed"])
                has_slangs.append(feats["has_slang"])
                has_romantics.append(feats["has_romantic"])
                word_counts.append(feats["word_count"])
                
                # Track statistics
                if source_info['source'] == 'cache':
                    cache_hits += 1
                elif source_info['source'] == 'inference':
                    inference_hits += 1
                elif source_info['source'] == 'hf':
                    hf_calls += 1
                
                # Log decision (if debug mode)
                if config.CSM_DEBUG_DECISIONS:
                    self.decision_logger.log_decision(
                        message_hash=self.nlp_cache.hash_text(text),
                        resolution_source=source_info['source'],
                        confidence_score=source_info.get('confidence', 1.0),
                        variance_factor=source_info.get('variance_factor', 1.0),
                        context_matches=source_info.get('context_matches', 0),
                        token_matches=source_info.get('token_matches', 0),
                        unknown_tokens=source_info.get('unknown_tokens', 0),
                        decision_reason=source_info.get('reason', ''),
                        total_time_ms=total_timer.elapsed_ms
                    )
            
            if (i + 1) % 50 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(texts)} - "
                    f"Cache: {cache_hits}, Inference: {inference_hits}, HF: {hf_calls}"
                )
        
        # Add scores to dataframe
        df_result['sentiment'] = [s if l=='positive' else -s for s,l in zip(sentiment_scores, sentiment_labels)] # Signed score
        df_result['sentiment_score'] = sentiment_scores # Confidence (0-1)
        df_result['sentiment_label'] = sentiment_labels
        df_result['toxicity_score'] = toxicity_scores
        df_result['toxicity_label'] = toxicity_labels
        df_result['combined_sentiment'] = combined_sentiments
        df_result['conflict_flag'] = conflict_flags
        df_result['teasing_flag'] = teasing_flags
        df_result['sentiment_uncertain'] = sentiment_uncertains
        
        df_result['emoji_count'] = emoji_counts
        df_result['emoji_valence'] = emoji_valences
        df_result['laughter_flag'] = laughter_flags
        df_result['code_mixed'] = code_mixeds
        df_result['has_slang'] = has_slangs
        df_result['has_romantic'] = has_romantics
        df_result['word_count'] = word_counts
        
        logger.info(
            f"CSM processing complete - Cache hits: {cache_hits}, "
            f"Inference: {inference_hits}, HF API calls: {hf_calls}"
        )
        
        # Calculate efficiency
        if len(texts) > 0:
            cache_hit_rate = (cache_hits / len(texts)) * 100
            hf_reduction = ((len(texts) - hf_calls) / len(texts)) * 100
            
            logger.info(
                f"CSM efficiency - Cache hit rate: {cache_hit_rate:.1f}%, "
                f"HF API reduction: {hf_reduction:.1f}%"
            )
        
        return df_result

    def _apply_fusion_rules(
        self, 
        sentiment: Dict[str, Any], 
        toxicity: Dict[str, Any], 
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply sensor fusion rules (copied from MessageProcessor for compatibility).
        """
        sent_score = sentiment["score"]
        sent_label = sentiment["label"]
        # Toxicity is already a score in our constructed dict
        tox_score = toxicity["score"]
        
        # Convert sentiment label to numeric score
        if "positive" in sent_label.lower():
            sentiment_numeric = sent_score
        elif "negative" in sent_label.lower():
            sentiment_numeric = -sent_score
        else:
            sentiment_numeric = 0.0
        
        # Conflict detection
        conflict = tox_score > config.TOXICITY_CONFLICT_THRESHOLD
        
        # Uncertainty detection
        uncertain = sent_score < config.SENTIMENT_CONFIDENCE_THRESHOLD
        
        # Teasing detection
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
        
        # Slang boost
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
    
    def _process_single_message(self, text: str) -> tuple:
        """
        Process a single message with CSM pipeline.
        
        Returns:
            (sentiment_score, toxicity_score, source_info_dict)
        """
        model_version_sent = config.CSM_SENTIMENT_MODEL_VERSION
        model_version_tox = config.CSM_TOXICITY_MODEL_VERSION
        
        # Step 1: Check cache
        cached = self.nlp_cache.get(
            text,
            config.SENTIMENT_MODEL,
            model_version_sent
        )
        
        if cached:
            return (
                cached['sentiment']['score'],
                cached['toxicity']['score'],
                {'source': 'cache', 'confidence': cached.get('confidence', 1.0)}
            )
        
        # Step 2: Run inference
        sent_inference = self.token_engine.infer_sentiment(text)
        tox_inference = self.token_engine.infer_toxicity(text)
        
        sentiment_score = sent_inference['score']
        toxicity_score = tox_inference['score']
        confidence = sent_inference['confidence']
        variance_factor = sent_inference['variance_factor']
        
        source_info = {
            'source': 'inference',
            'confidence': confidence,
            'variance_factor': variance_factor,
            'context_matches': sent_inference.get('context_matches', 0),
            'token_matches': sent_inference.get('token_matches', 0),
            'unknown_tokens': sent_inference.get('unknown_tokens', 0),
            'reason': f'Inference (conf={confidence:.2f}, var_factor={variance_factor:.2f})'
        }
        
        # Step 3: Decide if HF API call is needed
        should_call_hf = False
        reason = ""
        
        if not config.LIVE_HF_ENABLED:
            reason = "HF API disabled - using inference"
        elif self.nlp_cache.is_hf_throttled():
            reason = "HF API throttled - using inference"
        elif confidence < config.CSM_CONFIDENCE_THRESHOLD:
            if self.nlp_cache.can_call_hf_api():
                should_call_hf = True
                reason = f"Low confidence ({confidence:.2f}) - calling HF API"
            else:
                reason = "HF rate limit reached - using inference"
        else:
            reason = f"High confidence ({confidence:.2f}) - skipping HF API"
        
        # Step 4: Call HF API if needed
        if should_call_hf and self.hf_client:
            try:
                hf_sentiment = self.hf_client.get_sentiment([text])[0]
                hf_toxicity = self.hf_client.get_toxicity([text])[0]
                
                # Convert HF labels to scores
                sentiment_score = self._convert_sentiment_label(hf_sentiment)
                toxicity_score = hf_toxicity.get('score', 0.0)
                
                source_info['source'] = 'hf'
                source_info['confidence'] = 1.0
                source_info['reason'] = reason
                
                logger.debug(f"HF API called: sent={sentiment_score:.2f}, tox={toxicity_score:.2f}")
                
            except Exception as e:
                logger.warning(f"HF API call failed: {e} - using inference")
                source_info['reason'] = f"HF API error - {e}"
        
        # Step 5: Cache the result
        self.nlp_cache.set(
            text=text,
            model_name=config.SENTIMENT_MODEL,
            model_version=model_version_sent,
            sentiment={'label': 'inferred', 'score': sentiment_score},
            toxicity={'label': 'inferred', 'score': toxicity_score},
            confidence=source_info['confidence'],
            source=source_info['source']
        )
        
        # Step 6: Enqueue async token update
        if source_info['source'] in ['hf', 'inference'] and not self.suppress_learning:
            enqueue_token_update(text, sentiment_score, toxicity_score)
        
        return sentiment_score, toxicity_score, source_info
    
    def _convert_sentiment_label(self, hf_result: Dict[str, Any]) -> float:
        """
        Convert HF sentiment label to score.
        
        HF returns: {"label": "positive/neutral/negative", "score": confidence}
        We need: -1 to 1 scale (or 0 to 1 depending on model)
        """
        label = hf_result.get('label', 'neutral').lower()
        score = hf_result.get('score', 0.5)
        
        # Map label to sentiment score
        if 'positive' in label or 'pos' in label:
            return score  # 0.5 to 1.0
        elif 'negative' in label or 'neg' in label:
            return -score  # -1.0 to -0.5
        else:  # neutral
            return 0.0
    
    def _process_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback to standard processing without CSM."""
        logger.info("Using standard message processing (CSM disabled)")
        
        from .message_processor import MessageProcessor
        processor = MessageProcessor(self.hf_client)
        return processor.process_messages(df)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CSM statistics."""
        if not self.use_csm:
            return {'csm_enabled': False}
        
        stats = {
            'csm_enabled': True,
            'cache_stats': self.nlp_cache.stats(),
            'token_coverage': self.token_engine.get_token_coverage(),
        }
        
        if config.CSM_DEBUG_DECISIONS:
            stats['decision_log'] = self.decision_logger.get_summary_stats(hours=24)
        
        return stats


if __name__ == "__main__":
    # Test CSM processor
    import pandas as pd
    
    test_df = pd.DataFrame({
        'text': [
            'I love you so much!',
            'This is great',
            'I hate this',
            'Hello world'
        ]
    })
    
    processor = CSMMessageProcessor()
    result = processor.process_messages(test_df)
    
    print(result[['text', 'sentiment', 'toxicity']])
    print("\nStats:", processor.get_stats())
