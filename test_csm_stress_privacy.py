"""
CSM System Validation - Part 2: Stress, Reliability, Privacy & Accuracy
"""

import sys
import time
import json
import sqlite3
import random
import string
import logging
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from chatrel import config
from chatrel.utils import NLPCache, TokenStatsEngine, DecisionLogger
from chatrel.csm_processor import CSMMessageProcessor

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("CSM_TEST")

print("="*80)
print("ChatREL v4 - CSM Validation Part 2")
print("="*80)

# Load previous results
try:
    with open('test_results_csm.json', 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    results = {
        'stress_tests': {},
        'reliability_tests': {},
        'privacy_tests': {},
        'accuracy_tests': {},
        'recommendations': []
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_random_message(length: int = 10) -> str:
    words = ["love", "hate", "good", "bad", "amazing", "terrible", "okay", "fine", "great", "awful"]
    return " ".join(random.choices(words, k=length))

def mock_hf_response(texts: List[str]) -> List[Dict[str, Any]]:
    """Mock HF API response based on keywords."""
    responses = []
    for text in texts:
        score = 0.5
        label = "neutral"
        
        if "love" in text or "good" in text or "amazing" in text:
            score = 0.9
            label = "positive"
        elif "hate" in text or "bad" in text or "terrible" in text:
            score = 0.9
            label = "negative"
            
        responses.append({'label': label, 'score': score})
    return responses

# ============================================================================
# STRESS & LOAD TESTS
# ============================================================================

print("[4/8] STRESS & LOAD TESTS")
print("-" * 80)

stress_results = []

# Test 1: High volume inference throughput
try:
    engine = TokenStatsEngine()
    
    # Pre-populate stats to ensure inference works
    # We need > CSM_MIN_TOKEN_COUNT (5) updates
    words = ["love", "hate", "good", "bad", "amazing", "terrible"]
    for word in words:
        sentiment = 0.9 if word in ["love", "good", "amazing"] else -0.9
        for _ in range(10):  # Update 10 times to exceed threshold
            engine.update_token_stats(word, sentiment, 0.1)
            
    # Run 1000 inferences
    start_time = time.time()
    count = 1000
    
    for _ in range(count):
        msg = generate_random_message()
        engine.infer_sentiment(msg)
        
    duration = time.time() - start_time
    avg_latency_ms = (duration / count) * 1000
    throughput = count / duration
    
    status = 'PASS' if avg_latency_ms < 10.0 else 'WARN'  # Expect <10ms for pure inference
    
    stress_results.append({
        'test': 'Inference Throughput',
        'metric': 'Latency',
        'value': f'{avg_latency_ms:.2f} ms',
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '⚠'} Inference Latency: {avg_latency_ms:.2f} ms ({throughput:.0f} msg/sec)")
    
except Exception as e:
    print(f"✗ Stress Test Failed: {e}")
    stress_results.append({'test': 'Inference Throughput', 'status': 'FAIL', 'error': str(e)})

# Test 2: Cache throughput
try:
    cache = NLPCache()
    
    # Populate cache
    msg = "cached message"
    cache.set(msg, "model", "1.0", {'label':'pos', 'score':0.9}, {'label':'tox', 'score':0.1})
    
    start_time = time.time()
    count = 1000
    hits = 0
    
    for _ in range(count):
        res = cache.get(msg, "model", "1.0")
        if res:
            hits += 1
            
    duration = time.time() - start_time
    avg_latency_ms = (duration / count) * 1000
    
    status = 'PASS' if avg_latency_ms < 5.0 else 'WARN'
    
    stress_results.append({
        'test': 'Cache Throughput',
        'metric': 'Latency',
        'value': f'{avg_latency_ms:.2f} ms',
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '⚠'} Cache Latency: {avg_latency_ms:.2f} ms")
    
except Exception as e:
    print(f"✗ Cache Stress Test Failed: {e}")

results['stress_tests'] = stress_results
print()

# ============================================================================
# RELIABILITY TESTS
# ============================================================================

print("[5/8] RELIABILITY TESTS")
print("-" * 80)

rel_results = []

# Test 1: Redis Fallback
# We know Redis is down/unavailable in this env, so we verify DB fallback works
try:
    engine = TokenStatsEngine(redis_url="redis://nonexistent:6379/0")
    
    # This should not crash, but log a warning and use DB
    result = engine.infer_sentiment("love is good")
    
    status = 'PASS' if result['score'] != 0 else 'FAIL' # Should have stats from previous step
    
    rel_results.append({
        'test': 'Redis Failure Fallback',
        'expected': 'Graceful degradation to DB',
        'actual': 'Worked' if status == 'PASS' else 'Failed/Zero Score',
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '✗'} Redis Fallback: {status}")
    
except Exception as e:
    print(f"✗ Redis Fallback Failed: {e}")
    rel_results.append({'test': 'Redis Fallback', 'status': 'FAIL', 'error': str(e)})

results['reliability_tests'] = rel_results
print()

# ============================================================================
# PRIVACY TESTS
# ============================================================================

print("[6/8] PRIVACY TESTS")
print("-" * 80)

priv_results = []

# Test 1: Privacy Mode Hashing
try:
    # Enable privacy mode temporarily
    original_privacy = config.CSM_PRIVACY_MODE
    config.CSM_PRIVACY_MODE = True
    
    cache = NLPCache()
    secret_msg = "My secret phone is 555-0199"
    
    cache.set(secret_msg, "model", "1.0", {'label':'pos', 'score':0.5}, {'label':'tox', 'score':0.0})
    
    # Check DB directly
    conn = sqlite3.connect('chatrel_csm.db')
    cursor = conn.execute("SELECT text_hash, text_sanitized FROM message_cache WHERE text_hash = ?", 
                         (cache.hash_text(secret_msg),))
    row = cursor.fetchone()
    conn.close()
    
    # Restore config
    config.CSM_PRIVACY_MODE = original_privacy
    
    if row:
        text_hash, text_sanitized = row
        # In privacy mode, text_sanitized should be NULL or empty, NOT the raw text
        is_private = text_sanitized is None
        
        status = 'PASS' if is_private else 'FAIL'
        priv_results.append({
            'test': 'Privacy Mode Storage',
            'expected': 'NULL text_sanitized',
            'actual': f'Stored: {text_sanitized}',
            'status': status
        })
        print(f"{'✓' if status == 'PASS' else '✗'} Privacy Mode: {status}")
    else:
        print("✗ Privacy Test: Message not found in DB")
        priv_results.append({'test': 'Privacy Mode', 'status': 'FAIL', 'error': 'Not found'})

except Exception as e:
    print(f"✗ Privacy Test Failed: {e}")
    priv_results.append({'test': 'Privacy Mode', 'status': 'FAIL', 'error': str(e)})

# Test 2: PII Masking (Standard Mode)
try:
    cache = NLPCache()
    pii_msg = "Call me at 555-123-4567 or email test@example.com"
    
    normalized = cache.normalize_text(pii_msg)
    
    # Should replace phone/email
    has_phone = "555" in normalized
    has_email = "test@example.com" in normalized
    
    status = 'PASS' if not has_phone and not has_email else 'FAIL'
    
    priv_results.append({
        'test': 'PII Masking',
        'expected': 'Masked phone/email',
        'actual': normalized,
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '✗'} PII Masking: {status}")
    
except Exception as e:
    print(f"✗ PII Masking Failed: {e}")

results['privacy_tests'] = priv_results
print()

# ============================================================================
# A/B ACCURACY TESTS
# ============================================================================

print("[7/8] A/B ACCURACY TESTS")
print("-" * 80)

acc_results = []

try:
    # Mock HF Client
    mock_client = MagicMock()
    mock_client.get_sentiment.side_effect = lambda texts: mock_hf_response(texts)
    mock_client.get_toxicity.side_effect = lambda texts: [{'label': 'neutral', 'score': 0.05}] * len(texts)
    
    # 1. Run with CSM Disabled
    config.CSM_ENABLED = False
    processor_std = CSMMessageProcessor(hf_client=mock_client, use_csm=False)
    
    test_msgs = pd.DataFrame({'text': ["I love this", "I hate this", "This is okay"]})
    import pandas as pd # Re-import to be safe
    
    start = time.time()
    res_std = processor_std.process_messages(test_msgs)
    time_std = time.time() - start
    
    # 2. Run with CSM Enabled (and warm stats)
    config.CSM_ENABLED = True
    config.LIVE_HF_ENABLED = False # Force inference
    
    processor_csm = CSMMessageProcessor(hf_client=mock_client, use_csm=True)
    
    start = time.time()
    res_csm = processor_csm.process_messages(test_msgs)
    time_csm = time.time() - start
    
    # Compare
    diffs = []
    for i in range(len(test_msgs)):
        s_std = res_std.iloc[i]['sentiment']
        s_csm = res_csm.iloc[i]['sentiment']
        diffs.append(abs(s_std - s_csm))
        
    avg_diff = sum(diffs) / len(diffs)
    
    status = 'PASS' if avg_diff < 0.2 else 'WARN' # Allow some deviation for inference
    
    acc_results.append({
        'test': 'Accuracy Comparison',
        'metric': 'Avg Score Deviation',
        'value': f'{avg_diff:.3f}',
        'status': status
    })
    
    acc_results.append({
        'test': 'Performance Comparison',
        'metric': 'Speedup',
        'value': f'{time_std/time_csm:.1f}x',
        'status': 'PASS' if time_csm < time_std else 'FAIL'
    })
    
    print(f"{'✓' if status == 'PASS' else '⚠'} Accuracy Deviation: {avg_diff:.3f}")
    print(f"✓ Speedup: {time_std/time_csm:.1f}x (Std: {time_std*1000:.1f}ms, CSM: {time_csm*1000:.1f}ms)")
    
except Exception as e:
    print(f"✗ A/B Test Failed: {e}")
    acc_results.append({'test': 'A/B Comparison', 'status': 'FAIL', 'error': str(e)})

results['accuracy_tests'] = acc_results
print()

# ============================================================================
# FINAL REPORT GENERATION
# ============================================================================

print("[8/8] GENERATING FINAL REPORT")
print("-" * 80)

# Calculate final readiness
failures = 0
warnings = 0

for cat in results.values():
    if isinstance(cat, list):
        for test in cat:
            if test.get('status') == 'FAIL': failures += 1
            if test.get('status') == 'WARN': warnings += 1
    elif isinstance(cat, dict):
        for k, v in cat.items():
            if isinstance(v, str) and 'FAIL' in v: failures += 1

if failures == 0:
    verdict = "READY FOR PRODUCTION"
    if warnings > 0: verdict = "READY FOR CANARY (With Warnings)"
else:
    verdict = "NEEDS FIXES"

results['final_verdict'] = verdict

# Save final results
with open('test_results_csm_final.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Final Verdict: {verdict}")
print(f"Failures: {failures}, Warnings: {warnings}")
print("Full results saved to test_results_csm_final.json")
print("="*80)
