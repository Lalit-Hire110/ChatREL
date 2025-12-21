"""
CSM System Validation Test Suite
Comprehensive testing of Contextual Sentiment Memory implementation
"""

import sys
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ChatREL v4 - CSM System Validation Test Suite")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Test results storage
results = {
    'environment': {},
    'smoke_tests': {},
    'functional_tests': {},
    'stress_tests': {},
    'reliability_tests': {},
    'privacy_tests': {},
    'accuracy_tests': {},
    'recommendations': []
}

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

print("[1/8] ENVIRONMENT VALIDATION")
print("-" * 80)

# Check imports
try:
    import redis as redis_lib
    results['environment']['redis_library'] = 'OK'
    print("✓ Redis library: INSTALLED")
except ImportError as e:
    results['environment']['redis_library'] = f'MISSING: {e}'
    print("✗ Redis library: MISSING")

try:
    import celery as celery_lib
    results['environment']['celery_library'] = 'OK'
    print("✓ Celery library: INSTALLED")
except ImportError as e:
    results['environment']['celery_library'] = f'MISSING: {e}'
    print("✗ Celery library: MISSING")

try:
    from chatrel import config
    results['environment']['config'] = 'OK'
    print("✓ Config module: LOADED")
    
    # Check CSM config
    csm_enabled = getattr(config, 'CSM_ENABLED', None)
    live_hf = getattr(config, 'LIVE_HF_ENABLED', None)
    conf_threshold = getattr(config, 'CSM_CONFIDENCE_THRESHOLD', None)
    
    print(f"  - CSM_ENABLED: {csm_enabled}")
    print(f"  - LIVE_HF_ENABLED: {live_hf}")
    print(f"  - CSM_CONFIDENCE_THRESHOLD: {conf_threshold}")
    print(f"  - CSM_VARIANCE_THRESHOLD: {getattr(config, 'CSM_VARIANCE_THRESHOLD', None)}")
    print(f"  - CSM_MIN_TOKEN_COUNT: {getattr(config, 'CSM_MIN_TOKEN_COUNT', None)}")
    
    results['environment']['csm_config'] = {
        'enabled': csm_enabled,
        'live_hf': live_hf,
        'threshold': conf_threshold
    }
except Exception as e:
    results['environment']['config'] = f'ERROR: {e}'
    print(f"✗ Config module: ERROR - {e}")

# Check database initialization
try:
    db_path = Path('chatrel_csm.db')
    
    # Initialize database from schema
    schema_path = Path('database/schema_csm.sql')
    if schema_path.exists():
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.commit()
        
        # Verify tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['message_cache', 'word_stats', 'token_context_stats', 
                          'decision_log', 'sync_status', 'csm_metadata']
        
        missing = set(expected_tables) - set(tables)
        
        if not missing:
            results['environment']['database'] = 'OK'
            print(f"✓ Database: INITIALIZED ({len(tables)} tables)")
            print(f"  Tables: {', '.join(tables[:6])}")
        else:
            results['environment']['database'] = f'INCOMPLETE: Missing {missing}'
            print(f"✗ Database: INCOMPLETE - Missing {missing}")
    else:
        results['environment']['database'] = 'SCHEMA_MISSING'
        print("✗ Database schema file not found")
        
except Exception as e:
    results['environment']['database'] = f'ERROR: {e}'
    print(f"✗ Database: ERROR - {e}")

# Check Redis connection (optional)
try:
    redis_client = redis_lib.from_url('redis://localhost:6379/1', decode_responses=True)
    redis_client.ping()
    results['environment']['redis_server'] = 'OK'
    print("✓ Redis server: CONNECTED")
except Exception as e:
    results['environment']['redis_server'] = f'UNAVAILABLE: {e}'
    print(f"⚠ Redis server: UNAVAILABLE (will use DB-only mode)")
    print(f"  Reason: {e}")

print()

# ============================================================================
# SMOKE TESTS
# ============================================================================

print("[2/8] SMOKE TESTS")
print("-" * 80)

smoke_pass = 0
smoke_total = 0

# Test 1: Import CSM modules
smoke_total += 1
try:
    from chatrel.utils import NLPCache, TokenStatsEngine, DecisionLogger
    print("✓ Test 1: CSM modules import successfully")
    results['smoke_tests']['module_import'] = 'PASS'
    smoke_pass += 1
except Exception as e:
    print(f"✗ Test 1: Module import failed - {e}")
    results['smoke_tests']['module_import'] = f'FAIL: {e}'

# Test 2: Initialize NLPCache
smoke_total += 1
try:
    cache = NLPCache()
    print("✓ Test 2: NLPCache initialization successful")
    results['smoke_tests']['cache_init'] = 'PASS'
    smoke_pass += 1
except Exception as e:
    print(f"✗ Test 2: NLPCache init failed - {e}")
    results['smoke_tests']['cache_init'] = f'FAIL: {e}'

# Test 3: Initialize TokenStatsEngine
smoke_total += 1
try:
    engine = TokenStatsEngine()
    print("✓ Test 3: TokenStatsEngine initialization successful")
    results['smoke_tests']['engine_init'] = 'PASS'
    smoke_pass += 1
except Exception as e:
    print(f"✗ Test 3: TokenStatsEngine init failed - {e}")
    results['smoke_tests']['engine_init'] = f'FAIL: {e}'

# Test 4: Celery fallback mode
smoke_total += 1
try:
    from chatrel.tasks import CELERY_AVAILABLE, enqueue_token_update
    print(f"✓ Test 4: Celery status: {'AVAILABLE' if CELERY_AVAILABLE else 'FALLBACK MODE'}")
    results['smoke_tests']['celery'] = 'PASS' if not CELERY_AVAILABLE else 'AVAILABLE'
    smoke_pass += 1
except Exception as e:
    print(f"✗ Test 4: Celery import failed - {e}")
    results['smoke_tests']['celery'] = f'FAIL: {e}'

# Test 5: CSM processor
smoke_total += 1
try:
    from chatrel.csm_processor import CSMMessageProcessor
    processor = CSMMessageProcessor(use_csm=True)
    print("✓ Test 5: CSMMessageProcessor initialization successful")
    results['smoke_tests']['csm_processor'] = 'PASS'
    smoke_pass += 1
except Exception as e:
    print(f"✗ Test 5: CSMMessageProcessor init failed - {e}")
    results['smoke_tests']['csm_processor'] = f'FAIL: {e}'

print(f"\nSmoke Tests: {smoke_pass}/{smoke_total} PASSED")
results['smoke_tests']['summary'] = f'{smoke_pass}/{smoke_total}'
print()

# ============================================================================
# FUNCTIONAL TESTS
# ============================================================================

print("[3/8] FUNCTIONAL TESTS")
print("-" * 80)

func_results = []

# Test: Text normalization
try:
    from chatrel.utils.nlp_cache import NLPCache
    cache = NLPCache()
    
    test_cases = [
        ("I LOVE YOU!!!", "i love you!!!"),
        ("Hello    World", "hello world"),
        ("Test\n\nNewlines", "test newlines"),
    ]
    
    all_pass = True
    for input_text, expected in test_cases:
        normalized = cache.normalize_text(input_text)
        if normalized != expected:
            all_pass = False
            break
    
    status = 'PASS' if all_pass else 'FAIL'
    func_results.append({
        'test': 'Text Normalization',
        'expected': 'Lowercase + whitespace collapse',
        'actual': status,
        'status': status
    })
    print(f"{'✓' if all_pass else '✗'} Text normalization: {status}")
    
except Exception as e:
    func_results.append({
        'test': 'Text Normalization',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Text normalization: ERROR - {e}")

# Test: Cache set/get
try:
    cache = NLPCache()
    
    test_text = "I love this framework!"
    cache.set(
        text=test_text,
        model_name="test_model",
        model_version="1.0",
        sentiment={'label': 'positive', 'score': 0.95},
        toxicity={'label': 'non-toxic', 'score': 0.05},
        confidence=1.0,
        source='hf'
    )
    
    result = cache.get(test_text, "test_model", "1.0")
    
    status = 'PASS' if result and result['sentiment']['score'] == 0.95 else 'FAIL'
    func_results.append({
        'test': 'Cache Set/Get',
        'expected': 'Retrieve exact cached result',
        'actual': f"Score: {result['sentiment']['score'] if result else 'None'}",
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '✗'} Cache set/get: {status}")
    
except Exception as e:
    func_results.append({
        'test': 'Cache Set/Get',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Cache set/get: ERROR - {e}")

# Test: Token statistics update
try:
    engine = TokenStatsEngine()
    
    # Update token stats
    engine.update_token_stats("love", 0.9, 0.1)
    engine.update_token_stats("love", 0.85, 0.1)
    engine.update_token_stats("love", 0.95, 0.05)
    
    # Check database
    conn = sqlite3.connect('chatrel_csm.db')
    cursor = conn.execute(
        "SELECT sentiment_mean, sentiment_count FROM word_stats WHERE token = 'love'"
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        mean, count = row
        expected_mean = (0.9 + 0.85 + 0.95) / 3
        status = 'PASS' if abs(mean - expected_mean) < 0.01 and count == 3 else 'FAIL'
        
        func_results.append({
            'test': 'Token Stats Update (Welford)',
            'expected': f'Mean≈{expected_mean:.2f}, Count=3',
            'actual': f'Mean={mean:.2f}, Count={count}',
            'status': status
        })
        print(f"{'✓' if status == 'PASS' else '✗'} Token stats (Welford): {status}")
    else:
        func_results.append({
            'test': 'Token Stats Update',
            'expected': 'Token stored',
            'actual': 'Token not found',
            'status': 'FAIL'
        })
        print("✗ Token stats: FAIL - Token not found")
        
except Exception as e:
    func_results.append({
        'test': 'Token Stats Update',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Token stats: ERROR - {e}")

# Test: Context signature generation
try:
    from chatrel.utils.token_stats import ContextSignatureGenerator
    
    tokens = ['not', 'good']
    sig = ContextSignatureGenerator.generate(tokens, 1)
    
    has_neg = 'NEG' in sig
    status = 'PASS' if has_neg else 'FAIL'
    
    func_results.append({
        'test': 'Context Signature (Negation)',
        'expected': 'NEG_good',
        'actual': sig,
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '✗'} Context signature: {status} (sig={sig})")
    
except Exception as e:
    func_results.append({
        'test': 'Context Signature',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Context signature: ERROR - {e}")

# Test: Inference engine
try:
    engine = TokenStatsEngine()
    
    # Populate some stats first
    engine.update_token_stats("amazing", 0.95, 0.05)
    engine.update_token_stats("amazing", 0.9, 0.1)
    engine.update_token_stats("terrible", -0.9, 0.8)
    engine.update_token_stats("terrible", -0.85, 0.75)
    
    # Test inference
    result_pos = engine.infer_sentiment("This is amazing!")
    result_neg = engine.infer_sentiment("This is terrible!")
    
    pos_correct = result_pos['score'] > 0
    neg_correct = result_neg['score'] < 0
    
    status = 'PASS' if pos_correct and neg_correct else 'FAIL'
    
    func_results.append({
        'test': 'Inference Engine (Sentiment)',
        'expected': 'Positive>0, Negative<0',
        'actual': f'Pos={result_pos["score"]:.2f}, Neg={result_neg["score"]:.2f}',
        'status': status
    })
    print(f"{'✓' if status == 'PASS' else '✗'} Inference engine: {status}")
    
except Exception as e:
    func_results.append({
        'test': 'Inference Engine',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Inference engine: ERROR - {e}")

# Test: Variance-aware confidence
try:
    # Simulate high variance token
    engine = TokenStatsEngine()
    
    # Add values with high variance
    for val in [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]:
        engine.update_token_stats("unstable", val, 0.1)
    
    # Check if marked as unstable
    conn = sqlite3.connect('chatrel_csm.db')
    cursor = conn.execute(
        "SELECT is_stable, sentiment_variance FROM word_stats WHERE token = 'unstable'"
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        is_stable, variance = row
        status = 'PASS' if is_stable == 0 and variance > 0.35 else 'FAIL'
        
        func_results.append({
            'test': 'Variance-Aware Stability Flag',
            'expected': 'is_stable=0, variance>0.35',
            'actual': f'is_stable={is_stable}, variance={variance:.3f}',
            'status': status
        })
        print(f"{'✓' if status == 'PASS' else '✗'} Variance stability: {status}")
    else:
        func_results.append({
            'test': 'Variance Stability',
            'expected': 'Token stored',
            'actual': 'Token not found',
            'status': 'FAIL'
        })
        print("✗ Variance stability: FAIL")
        
except Exception as e:
    func_results.append({
        'test': 'Variance Stability',
        'expected': 'PASS',
        'actual': f'ERROR: {e}',
        'status': 'FAIL'
    })
    print(f"✗ Variance stability: ERROR - {e}")

results['functional_tests'] = func_results
print(f"\nFunctional Tests: {sum(1 for r in func_results if r['status'] == 'PASS')}/{len(func_results)} PASSED")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("[8/8] SAVING TEST RESULTS")
print("-" * 80)

# Save to JSON
results_file = Path('test_results_csm.json')
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {results_file}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Smoke Tests: {results['smoke_tests'].get('summary', 'N/A')}")
print(f"Functional Tests: {sum(1 for r in func_results if r['status'] == 'PASS')}/{len(func_results)} PASSED")
print()
print("Environment Status:")
print(f"  - Database: {results['environment'].get('database', 'UNKNOWN')}")
print(f"  - Redis: {results['environment'].get('redis_server', 'UNKNOWN')}")
print(f"  - Celery: {results['environment'].get('celery', 'UNKNOWN')}")
print()
print("Completed:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("="*80)
