"""
CSM System Final Validation Suite
Comprehensive testing of Contextual Sentiment Memory implementation
"""

import sys
import sqlite3
import json
import time
import os
import random
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from chatrel import config
from chatrel.utils import NLPCache, TokenStatsEngine, DecisionLogger
from chatrel.csm_processor import CSMMessageProcessor

# Configure logging
logging.basicConfig(level=logging.CRITICAL) # Suppress logs for clean output
logger = logging.getLogger("CSM_TEST")

print("="*80)
print("ChatREL v4 - CSM Final Validation Suite")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = {
    'environment': {},
    'smoke_tests': {},
    'functional_tests': [],
    'stress_tests': [],
    'reliability_tests': [],
    'privacy_tests': [],
    'accuracy_tests': [],
    'final_verdict': 'UNKNOWN'
}

# ============================================================================
# 0. SETUP
# ============================================================================
print("\n[0/7] SETUP")
if os.path.exists('chatrel_csm.db'):
    try:
        os.remove('chatrel_csm.db')
        print("✓ Removed existing database")
    except Exception as e:
        print(f"⚠ Could not remove database: {e}")

# Initialize DB schema
try:
    cache = NLPCache() # This initializes DB
    print("✓ Database initialized via NLPCache")
except Exception as e:
    print(f"✗ Database init failed: {e}")
    sys.exit(1)

# ============================================================================
# 1. ENVIRONMENT & SMOKE
# ============================================================================
print("\n[1/7] ENVIRONMENT & SMOKE")

# Redis
try:
    import redis
    results['environment']['redis_lib'] = 'OK'
    try:
        r = redis.from_url(config.CSM_REDIS_URL)
        r.ping()
        results['environment']['redis_server'] = 'CONNECTED'
        print("✓ Redis: CONNECTED")
    except Exception as e:
        results['environment']['redis_server'] = f'UNAVAILABLE ({e})'
        print(f"⚠ Redis: UNAVAILABLE (Using DB-only mode)")
except ImportError:
    results['environment']['redis_lib'] = 'MISSING'
    print("✗ Redis lib: MISSING")

# Celery
try:
    import celery
    results['environment']['celery_lib'] = 'OK'
    print("✓ Celery: INSTALLED")
except ImportError:
    results['environment']['celery_lib'] = 'MISSING'
    print("✗ Celery: MISSING")

# Config
print(f"  CSM Enabled: {config.CSM_ENABLED}")
print(f"  Live HF: {config.LIVE_HF_ENABLED}")

# ============================================================================
# 2. FUNCTIONAL TESTS
# ============================================================================
print("\n[2/7] FUNCTIONAL TESTS")

def run_test(category, name, expected, actual_fn):
    try:
        actual, status = actual_fn()
        results[category].append({
            'test': name,
            'expected': expected,
            'actual': str(actual),
            'status': status
        })
        symbol = '✓' if status == 'PASS' else '✗'
        print(f"{symbol} {name}: {status}")
    except Exception as e:
        results[category].append({
            'test': name,
            'expected': expected,
            'actual': f"ERROR: {e}",
            'status': 'FAIL'
        })
        print(f"✗ {name}: ERROR - {e}")

# Test: Normalization
run_test('functional_tests', 'Text Normalization', 'normalized', lambda: (
    NLPCache().normalize_text("  HeLLo   World!  "), 
    'PASS' if NLPCache().normalize_text("  HeLLo   World!  ") == "hello world!" else 'FAIL'
))

# Test: Cache
def test_cache():
    c = NLPCache()
    c.set("test msg", "m1", "v1", {'label':'pos','score':0.9}, {'label':'tox','score':0.0})
    res = c.get("test msg", "m1", "v1")
    return res['sentiment']['score'], 'PASS' if res and res['sentiment']['score'] == 0.9 else 'FAIL'
run_test('functional_tests', 'Cache Set/Get', '0.9', test_cache)

# Test: Stats Update & Inference
def test_inference():
    e = TokenStatsEngine()
    # Train "good"
    for _ in range(10): e.update_token_stats("good", 0.9, 0.0)
    # Train "bad"
    for _ in range(10): e.update_token_stats("bad", -0.9, 0.0)
    
    res = e.infer_sentiment("good bad")
    # Should be near 0
    score = res['score']
    status = 'PASS' if -0.2 <= score <= 0.2 else 'FAIL'
    return f"{score:.2f}", status
run_test('functional_tests', 'Inference Logic', 'Near 0.0', test_inference)

# Test: Variance
def test_variance():
    e = TokenStatsEngine()
    # High variance input [-1, 1, -1, 1...]
    for _ in range(10):
        e.update_token_stats("unstable", 1.0, 0.0)
        e.update_token_stats("unstable", -1.0, 0.0)
    
    with sqlite3.connect('chatrel_csm.db') as conn:
        row = conn.execute("SELECT sentiment_variance, is_stable FROM word_stats WHERE token='unstable'").fetchone()
    
    var = row[0]
    stable = row[1]
    # Variance of [-1, 1] is 1.0. Threshold is 0.35. Should be unstable (0).
    status = 'PASS' if var > 0.35 and stable == 0 else 'FAIL'
    return f"Var={var:.2f}, Stable={stable}", status
run_test('functional_tests', 'Variance Detection', 'Var>0.35, Stable=0', test_variance)

# ============================================================================
# 3. STRESS TESTS
# ============================================================================
print("\n[3/7] STRESS TESTS")

def test_latency():
    e = TokenStatsEngine()
    start = time.time()
    count = 500
    for _ in range(count):
        e.infer_sentiment("good bad good bad")
    duration = time.time() - start
    avg = (duration / count) * 1000
    status = 'PASS' if avg < 15.0 else 'WARN'
    return f"{avg:.2f} ms", status
run_test('stress_tests', 'Inference Latency (500 ops)', '<15ms', test_latency)

# ============================================================================
# 4. RELIABILITY TESTS
# ============================================================================
print("\n[4/7] RELIABILITY TESTS")

def test_redis_fallback():
    # Force bad redis URL
    e = TokenStatsEngine(redis_url="redis://nonexistent:6379")
    # Should not crash
    res = e.infer_sentiment("good")
    status = 'PASS' if res else 'FAIL'
    return "Handled", status
run_test('reliability_tests', 'Redis Connection Failure', 'Handled', test_redis_fallback)

# ============================================================================
# 5. PRIVACY TESTS
# ============================================================================
print("\n[5/7] PRIVACY TESTS")

def test_privacy_mode():
    orig = config.CSM_PRIVACY_MODE
    config.CSM_PRIVACY_MODE = True
    c = NLPCache()
    c.set("secret message", "m", "v", {}, {})
    
    with sqlite3.connect('chatrel_csm.db') as conn:
        row = conn.execute("SELECT text_sanitized FROM message_cache WHERE text_sanitized IS NOT NULL").fetchone()
    
    config.CSM_PRIVACY_MODE = orig
    status = 'PASS' if row is None else 'FAIL'
    return "No raw text found", status
run_test('privacy_tests', 'Privacy Mode Storage', 'No raw text', test_privacy_mode)

def test_pii_masking():
    c = NLPCache()
    masked = c.sanitize_text("Call 5551234567")
    status = 'PASS' if "[PHONE]" in masked else 'FAIL'
    return masked, status
run_test('privacy_tests', 'PII Masking', 'Contains [PHONE]', test_pii_masking)

# ============================================================================
# 6. ACCURACY TESTS (A/B)
# ============================================================================
print("\n[6/7] ACCURACY TESTS")

def test_ab_accuracy():
    # Mock HF
    mock = MagicMock()
    mock.get_sentiment.return_value = [{'label':'pos', 'score':0.9}]
    mock.get_toxicity.return_value = [{'label':'neu', 'score':0.0}]
    
    df = pd.DataFrame({'text': ['good job']})
    
    # Std
    config.CSM_ENABLED = False
    p1 = CSMMessageProcessor(hf_client=mock, use_csm=False)
    # Use combined_sentiment which is the final output for both processors
    r1 = p1.process_messages(df).iloc[0]['combined_sentiment']
    
    # CSM (Inference)
    config.CSM_ENABLED = True
    config.LIVE_HF_ENABLED = False
    # Ensure stats exist
    e = TokenStatsEngine()
    for _ in range(10): e.update_token_stats("good", 0.9, 0.0)
    for _ in range(10): e.update_token_stats("job", 0.5, 0.0)
    
    p2 = CSMMessageProcessor(hf_client=mock, use_csm=True)
    r2 = p2.process_messages(df).iloc[0]['combined_sentiment']
    
    diff = abs(r1 - r2)
    status = 'PASS' if diff < 0.3 else 'WARN'
    return f"Diff={diff:.2f}", status

run_test('accuracy_tests', 'A/B Score Deviation', 'Diff < 0.3', test_ab_accuracy)

# ============================================================================
# 7. REPORT
# ============================================================================
print("\n[7/7] FINAL VERDICT")

failures = 0
warnings = 0
for cat in results.values():
    if isinstance(cat, list):
        for t in cat:
            if t['status'] == 'FAIL': failures += 1
            if t['status'] == 'WARN': warnings += 1

if failures == 0:
    verdict = "READY FOR PRODUCTION"
    if warnings > 0: verdict = "READY FOR CANARY (With Warnings)"
else:
    verdict = "NEEDS FIXES"

results['final_verdict'] = verdict
print(f"Verdict: {verdict}")
print(f"Failures: {failures}, Warnings: {warnings}")

with open('test_results_csm_final.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to test_results_csm_final.json")
print("="*80)
