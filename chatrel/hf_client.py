"""
HuggingFace Inference API Client for ChatREL v4
Handles batching, rate limiting, retries, and caching
"""

import time
import logging
import json
from typing import List, Dict, Any, Optional
import requests
import concurrent.futures

from . import config
from .cache import ResponseCache

logger = logging.getLogger(__name__)


class HFClient:
    """
    Client for HuggingFace Inference API with batching and caching.
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        batch_size: Optional[int] = None,
        rate_limit_sleep: Optional[float] = None,
        use_cache: bool = True,
        mock_mode: bool = False,
    ):
        """
        Initialize HF client.
        
        Args:
            token: HF API token (default from config)
            batch_size: Number of texts per batch (default from config)
            rate_limit_sleep: Sleep between batches in seconds (default from config)
            use_cache: Whether to use cache (default True)
            mock_mode: Use mock responses for testing (default False)
        """
        self.token = token or config.HF_TOKEN
        if not self.token and not mock_mode:
            raise ValueError("HF_TOKEN not set - add to .env file or pass as argument")
        
        self.batch_size = batch_size or config.BATCH_SIZE
        self.rate_limit_sleep = rate_limit_sleep or config.RATE_LIMIT_SLEEP
        self.timeout = config.API_TIMEOUT
        self.max_retries = config.MAX_RETRIES
        
        self.use_cache = use_cache
        self.cache = ResponseCache() if use_cache else None
        
        self.mock_mode = mock_mode
        self.mock_responses = self._load_mock_responses() if mock_mode else {}
        
        logger.info(f"HFClient initialized (batch_size={self.batch_size}, cache={use_cache}, mock={mock_mode})")
    
    def _load_mock_responses(self) -> Dict[str, Any]:
        """Load mock responses from sample_data/mock_responses.json."""
        mock_path = config.PROJECT_ROOT / "sample_data" / "mock_responses.json"
        if mock_path.exists():
            with open(mock_path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"Mock responses file not found: {mock_path}")
        return {}
    
    def query(self, texts: List[str], model_name: str) -> List[Dict[str, Any]]:
        """
        Query HuggingFace API for multiple texts with batching.
        
        Args:
            texts: List of text inputs
            model_name: HF model identifier
        
        Returns:
            List of response dicts (one per input text)
        """
        if not texts:
            return []
        
        results = [None] * len(texts)
        batches = []
        
        # Prepare batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batches.append((i, batch_texts))
            
        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches (concurrent={config.HF_CONCURRENT_REQUESTS})")
        
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.HF_CONCURRENT_REQUESTS) as executor:
            future_to_batch = {
                executor.submit(self._query_batch, batch_texts, model_name): (i, batch_texts)
                for i, batch_texts in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                start_idx, batch_texts = future_to_batch[future]
                try:
                    batch_results = future.result()
                    # Place results in correct order
                    for j, res in enumerate(batch_results):
                        results[start_idx + j] = res
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Fill with fallbacks
                    for j in range(len(batch_texts)):
                        results[start_idx + j] = self._fallback_response(batch_texts[j], model_name)
        
        return results
    
    def _query_batch(self, texts: List[str], model_name: str) -> List[Dict[str, Any]]:
        """Query a single batch of texts."""
        results = []
        
        for text in texts:
            # Check cache first
            if self.use_cache and self.cache:
                cached = self.cache.get(model_name, text)
                if cached is not None:
                    results.append(cached)
                    continue
            
            # Query API or mock
            if self.mock_mode:
                response = self._mock_query(text, model_name)
            else:
                response = self._api_query(text, model_name)
            
            # Cache response
            if self.use_cache and self.cache and response:
                self.cache.set(model_name, text, response)
            
            results.append(response)
        
        return results
    
    def _api_query(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Query HuggingFace API for a single text with retries.
        """
        # Updated to new router-based endpoint (old api-inference endpoint deprecated)
        url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": text}
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Normalize response format
                    return self._normalize_response(result, model_name)
                
                elif response.status_code == 410:
                    # Endpoint deprecated - this is a configuration error, not a temporary failure
                    error_msg = f"API endpoint deprecated (410): {response.text[:200]}"
                    logger.error(error_msg)
                    logger.error("The HuggingFace API endpoint has changed. Please update HF_API_BASE in hf_client.py")
                    raise ValueError(error_msg)
                
                elif response.status_code == 404:
                    # Model not found - this is a configuration error
                    error_msg = f"Model not found (404): {model_name}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    last_error = f"Server error: {response.status_code}"
                    continue
                
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    last_error = "Rate limited"
                    continue
                
                else:
                    # Other client errors (400, 401, 403, etc.)
                    logger.error(f"API error {response.status_code}: {response.text[:200]}")
                    # For temporary issues like model loading, use fallback
                    if response.status_code == 503:
                        logger.warning("Model is loading, using fallback response")
                        return self._fallback_response(text, model_name)
                    # For other client errors, retry might help
                    last_error = f"Client error: {response.status_code}"
                    continue
            
            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt+1}/{self.max_retries}")
                last_error = "Timeout"
                continue
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                last_error = str(e)
                break
        
        # All retries failed - use fallback only for temporary failures
        logger.error(f"All retries failed for {model_name}: {last_error}")
        return self._fallback_response(text, model_name)
    
    def _mock_query(self, text: str, model_name: str) -> Dict[str, Any]:
        """Return mock response for testing."""
        # Check if specific mock exists
        if text in self.mock_responses:
            return self.mock_responses[text]
        
        # Generic mock based on model type
        if "sentiment" in model_name.lower():
            return {"label": "neutral", "score": 0.5}
        elif "toxicity" in model_name.lower() or "toxic" in model_name.lower():
            return {"label": "non-toxic", "score": 0.1}
        else:
            return {"label": "unknown", "score": 0.5}
    
    def _normalize_response(self, result: Any, model_name: str) -> Dict[str, Any]:
        """
        Normalize different HF response formats to consistent structure.
        
        Sentiment model returns: [[{"label": "...", "score": ...}, ...]]
        Toxicity model returns: [[{"label": "...", "score": ...}]]
        """
        # Handle list wrapping
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                # Double-nested list [[{...}]]
                result = result[0]
            
            if isinstance(result[0], dict):
                # Take highest score entry
                best = max(result, key=lambda x: x.get("score", 0))
                return {
                    "label": best.get("label", "unknown"),
                    "score": best.get("score", 0.5),
                    "all_labels": result if len(result) > 1 else None,
                }
        
        # Fallback
        if isinstance(result, dict):
            return {
                "label": result.get("label", "unknown"),
                "score": result.get("score", 0.5),
            }
        
        logger.warning(f"Unexpected response format from {model_name}: {result}")
        return self._fallback_response("", model_name)
    
    def _fallback_response(self, text: str, model_name: str) -> Dict[str, Any]:
        """Return neutral fallback when API fails."""
        if "sentiment" in model_name.lower():
            return {"label": "neutral", "score": 0.5}
        elif "toxicity" in model_name.lower() or "toxic" in model_name.lower():
            return {"label": "non-toxic", "score": 0.1}
        else:
            return {"label": "unknown", "score": 0.5}
    
    def get_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Convenience method for sentiment analysis."""
        return self.query(texts, config.SENTIMENT_MODEL)
    
    def get_toxicity(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Convenience method for toxicity detection."""
        return self.query(texts, config.TOXICITY_MODEL)


if __name__ == "__main__":
    # Test in mock mode
    client = HFClient(mock_mode=True)
    
    texts = ["I love you!", "You are terrible!", "Hello world"]
    
    print("Testing sentiment:")
    sentiments = client.get_sentiment(texts)
    for text, sent in zip(texts, sentiments):
        print(f"  {text[:30]}: {sent}")
    
    print("\nTesting toxicity:")
    toxicities = client.get_toxicity(texts)
    for text, tox in zip(texts, toxicities):
        print(f"  {text[:30]}: {tox}")
    
    if client.cache:
        print("\nCache stats:", json.dumps(client.cache.stats(), indent=2))
