# ChatREL v4 - Contextual Sentiment Memory (CSM) Guide

## Overview

Contextual Sentiment Memory (CSM) is an advanced caching and learning system that dramatically reduces API costs and improves analysis speed by:

- **Caching analyzed messages** for instant reuse
- **Learning token-level sentiment patterns** over time
- **Inferring sentiment** for new messages using statistical models
- **Adapting to conversation style** through continuous learning

## Key Features

### ✅ Message-Level Caching
- Exact message matching with hash-based lookup
- Version-aware cache invalidation
- Redis hot cache + SQLite persistence
- Privacy mode (store only hashes)

### ✅ Token-Level Statistics
- Welford's algorithm for incremental mean/variance
- Per-token sentiment and toxicity tracking
- Emoji lexicon integration
- Min sample thresholds for reliability

### ✅ Context-Aware Scoring
- Negation detection ("not good" vs "good")
- Previous/next token patterns
- Emoji presence indicators
- Capitalization and punctuation context

### ✅ Background Learning
- Async worker (Celery) for non-blocking updates
- Periodic Redis → Database sync
- Automatic decision log cleanup

### ✅ Variance-Aware Confidence
- Statistical stability thresholds
- Confidence adjustment based on token variance
- Unstable token detection and filtering

### ✅ HF API Rate Limiting
- Rolling window throttle
- Automatic fallback to inference when throttled
- Configurable limits

### ✅ Decision Logging
- Debug mode trace logging
- Performance metrics (latency, confidence)
- Resolution source tracking

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `redis` - For hot cache
- `celery` - For background workers
- `emoji` - For emoji detection

### 2. Start the Stack (Recommended)

The easiest way to run ChatREL v4 with full CSM capabilities is using Docker Compose.

**Linux/Mac:**
```bash
./scripts/start_stack.sh
```

**Windows (PowerShell):**
```powershell
./scripts/start_stack.ps1
```

This will start:
- Redis (port 6379)
- Web Application (port 5000)
- Celery Worker (background learning)
- Celery Beat (periodic sync)

**Manual Docker Compose:**
```bash
docker-compose up -d
```

### 3. Manual Setup (Without Docker)

If you cannot use Docker, you must install Redis manually:

**Windows:**
- Install [Memurai](https://www.memurai.com/) (Redis-compatible for Windows)
- Or use WSL2 to install Redis

**Linux/Mac:**
```bash
brew install redis
redis-server
```

Then start the components separately:

1. Start Redis
2. Start Web App: `python -m chatrel.api`
3. Start Celery Worker: `celery -A chatrel.tasks.celery_app worker --loglevel=info`
4. Start Celery Beat: `celery -A chatrel.tasks.celery_app beat --loglevel=info`

### 3. Initialize Database

The CSM database is automatically created on first run. To manually initialize:

```bash
sqlite3 chatrel_csm.db < database/schema_csm.sql
```

### 4. Configure Environment

Copy `.env.example` to `.env` and configure CSM settings:

```bash
# Enable CSM
CSM_ENABLED=True

# Disable HF API for inference-only mode
LIVE_HF_ENABLED=False

# Adjust confidence threshold
CSM_CONFIDENCE_THRESHOLD=0.70
```

### 5. (Optional) Start Celery Worker

For async learning:

```bash
celery -A chatrel.tasks.update_word_stats worker --loglevel=info
```

For periodic tasks (Redis sync):

```bash
celery -A chatrel.tasks.update_word_stats beat --loglevel=info
```

---

## Usage

### Basic Usage

CSM is automatically integrated into the analysis pipeline:

```python
from chatrel.analysis_engine import run_analysis
from pathlib import Path

# Run analysis with CSM enabled (default)
result = run_analysis(
    filepath=Path("chat.txt"),
    use_nlp=True,
    use_cache=True  # Enables both HF cache and CSM
)
```

### Warm-up Prefill

Before using CSM in production, run the warm-up script to populate token statistics:

```bash
python scripts/warmup_prefill.py --data-dir ./sample_data --limit 10
```

This generates:
- `warmup_reports/warmup_report_YYYYMMDD_HHMMSS.md` - Detailed analysis
- `warmup_reports/warmup_data_YYYYMMDD_HHMMSS.json` - Machine-readable data

### Checking CSM Stats

```python
from chatrel.csm_processor import CSMMessageProcessor

processor = CSMMessageProcessor()
stats = processor.get_stats()

print(f"Cache hit rate: {stats['cache_stats']['total_cache_hits']}")
print(f"Token coverage: {stats['token_coverage']['coverage_ratio']:.1%}")
```

---

## Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CSM_ENABLED` | `True` | Enable CSM system |
| `LIVE_HF_ENABLED` | `False` | Call HF API for low-confidence messages |
| `CSM_CONFIDENCE_THRESHOLD` | `0.70` | Min confidence to skip HF API |

### Variance & Stability

| Variable | Default | Description |
|----------|---------|-------------|
| `CSM_VAR_CONFIDENCE_WEIGHT` | `1.0` | Weight of variance penalty |
| `CSM_VARIANCE_THRESHOLD` | `0.35` | Max variance before marking unstable |

### Token Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `CSM_MIN_TOKEN_COUNT` | `5` | Min samples for global token stats |
| `CSM_MIN_CONTEXT_COUNT` | `3` | Min samples for context-specific stats |

### Redis & Database

| Variable | Default | Description |
|----------|---------|-------------|
| `CSM_REDIS_URL` | `redis://localhost:6379/1` | Redis connection |
| `CSM_REDIS_TTL_HOURS` | `24` | Redis entry TTL |
| `CSM_REDIS_SYNC_INTERVAL_HOURS` | `6` | Sync frequency |
| `CSM_DB_PATH` | `chatrel_csm.db` | SQLite database path |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_HF_CALLS_PER_MIN` | `50` | Max HF API calls per minute |
| `HF_THROTTLE_WINDOW_SECONDS` | `60` | Rate limit window |

### Privacy & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `CSM_PRIVACY_MODE` | `False` | Store only hashes |
| `CSM_DEBUG_DECISIONS` | `False` | Enable decision logging |

---

## Architecture

### Data Flow

```
Message Input
    ↓
Text Normalization
    ↓
Cache Lookup (Redis → DB)
    ↓
├─ Cache Hit → Return Result
└─ Cache Miss
    ↓
Token-Level Inference
    ↓
Context Signature Matching
    ↓
Confidence Calculation
    ├─ Variance Adjustment
    └─ Threshold Check
        ↓
    ├─ High Confidence → Use Inference
    └─ Low Confidence
        ├─ HF API Enabled → Call API
        └─ HF API Disabled → Use Inference
            ↓
        Cache Result
            ↓
        Async Token Update (Celery)
```

### Database Schema

**message_cache**
- Exact message → sentiment/toxicity mapping
- Version-aware (model version tracking)
- Privacy mode support (hash only)

**word_stats**
- Global token statistics
- Welford mean/variance
- Emoji boost values
- Stability flags

**token_context_stats**
- Context-specific patterns
- Negation, emoji, capitalization
- Higher precision than global stats

**decision_log** (debug mode only)
- Resolution source tracking
- Confidence and variance factors
- Performance metrics

---

## Performance Benchmarks

### Expected Performance (After Warm-up)

| Scenario | Latency | HF API Calls |
|----------|---------|--------------|
| Cache hit | 20-70ms | 0 |
| Inference (no HF) | 100-250ms | 0 |
| HF API call | 1,500-3,000ms | 1 per message |

### Efficiency Metrics

With good warm-up coverage (>70% token coverage):

- **Cache hit rate**: 60-80%
- **HF API reduction**: 85-95%
- **Cost savings**: 90%+ (based on HF API pricing)
- **Throughput increase**: 5-10x

---

## Operational Modes

Mode | Description | Use Case
|----|-------------|---------|
| **Inference-Only** | `LIVE_HF_ENABLED=False` | Production after warm-up, cost-sensitive |
| **Hybrid** | `LIVE_HF_ENABLED=True`, threshold=0.70 | Balanced accuracy & cost |
| **HF-First** | `CSM_ENABLED=False` | Highest accuracy, highest cost |
| **Debug** | `CSM_DEBUG_DECISIONS=True` | Development, tuning thresholds |

---

## Troubleshooting

### Issue: Low Cache Hit Rate

**Solution**:
- Run warm-up prefill on more historical data
- Check if message text is normalized consistently
- Verify model version hasn't changed

### Issue: High HF Dependency (>30%)

**Solution**:
- Lower `CSM_CONFIDENCE_THRESHOLD` (e.g., 0.60)
- Process more diverse training data in warm-up
- Check for high-variance tokens in warm-up report

### Issue: Celery Workers Not Running

**Symptom**: Synchronous updates, slow performance

**Solution**:
```bash
# Check worker status
celery -A chatrel.tasks.update_word_stats inspect active

# Restart workers
celery -A chatrel.tasks.update_word_stats worker --loglevel=info
```

### Issue: Redis Connection Failed

**Solution**:
- Check Redis is running: `redis-cli ping`
- Verify `CSM_REDIS_URL` in `.env`
- CSM will fallback to DB-only mode (slower)

### Issue: High Variance Tokens

**Symptom**: Many unstable tokens in warm-up report

**Solution**:
- Increase `CSM_MIN_TOKEN_COUNT` to require more samples
- Review token quality (slang, typos reduce stability)
- Increase `CSM_VARIANCE_THRESHOLD` to be more permissive

---

## Best Practices

### 1. Warm-up Before Production

Always run warm-up prefill on representative historical data:

```bash
python scripts/warmup_prefill.py --data-dir ./historical_chats --limit 50
```

Review the report to ensure:
- Token coverage > 70%
- HF dependency < 20%
- High-variance tokens < 100

### 2. Monitor Decision Logs

Enable debug mode periodically:

```bash
CSM_DEBUG_DECISIONS=True
```

Query decision_log table to analyze:
- Confidence distribution
- HF API call patterns
- Latency percentiles

### 3. Tune Confidence Threshold

Start conservative (0.70), then adjust based on:
- Accuracy requirements (higher = more HF calls)
- Cost constraints (lower = fewer HF calls)
- Warm-up coverage (better coverage = lower threshold)

### 4. Regular Redis Sync

Monitor sync status:

```sql
SELECT * FROM sync_status;
```

If errors > 0, investigate Redis connectivity.

### 5. Privacy Mode for Sensitive Data

Enable for PII-sensitive conversations:

```bash
CSM_PRIVACY_MODE=True
```

This stores only hashes, making data unrecoverable.

---

## API Reference

### CSMMessageProcessor

```python
from chatrel.csm_processor import CSMMessageProcessor

processor = CSMMessageProcessor(
    hf_client=None,  # Optional HF client
    use_csm=True     # Enable CSM (default from config)
)

# Process messages
df_result = processor.process_messages(df)

# Get stats
stats = processor.get_stats()
```

### NLPCache

```python
from chatrel.utils import NLPCache

cache = NLPCache()

# Get cached result
result = cache.get(text, model_name, model_version)

# Set cache entry
cache.set(text, model_name, model_version, sentiment, toxicity)

# Stats
stats = cache.stats()
```

### TokenStatsEngine

```python
from chatrel.utils import TokenStatsEngine

engine = TokenStatsEngine()

# Infer sentiment
result = engine.infer_sentiment("I love this!")
# Returns: {'score': float, 'confidence': float, 'variance_factor': float}

# Get coverage
coverage = engine.get_token_coverage()
```

---

## Maintenance

### Weekly Tasks

- Check decision log summary
- Review cache hit rates
- Verify Celery workers are running

### Monthly Tasks

- Run warm-up on new data to learn new patterns
- Vacuum database: `sqlite3 chatrel_csm.db "VACUUM;"`
- Clean old decision logs (auto-cleanup runs daily)

### Quarterly Tasks

- Review high-variance tokens
- Tune confidence threshold based on accuracy metrics
- Update model version if HF models change

---

## FAQ

**Q: Does CSM work without Redis?**
A: Yes, but performance is degraded. Redis provides fast cache lookups.

**Q: Can I use Postgres instead of SQLite?**
A: Yes, update `CSM_DB_PATH` to a PostgreSQL connection string (requires SQLAlchemy).

**Q: How much space does CSM use?**
A: ~1MB per 1,000 cached messages, ~500KB per 10,000 unique tokens.

**Q: Can I disable CSM temporarily?**
A: Yes, set `CSM_ENABLED=False`. System reverts to standard HF API calls.

**Q: What happens if warm-up data is biased?**
A: Token statistics will reflect that bias. Use diverse, representative data.

---

## Support

For issues or questions:
- Check troubleshooting section above
- Review decision logs if debug mode is enabled
- Examine warm-up reports for coverage gaps
- Verify configuration in `.env` matches requirements

---

*Last updated: 2025-11-29*

---

## Demo Mode & Runtime Switching

ChatREL v4 supports three runtime modes selectable via the UI or configuration:

### 1. Normal Mode
- **Behavior**: Standard analysis using configured settings.
- **CSM**: Uses `CSM_ENABLED` from `.env`.
- **Use Case**: Default production usage.

### 2. CSM Mode
- **Behavior**: Forces Contextual Sentiment Memory (CSM) to be **ENABLED** for the session.
- **Overrides**: Ignores `CSM_ENABLED=False` in config.
- **Use Case**: Testing CSM learning/inference without changing global config.

### 3. Demo Mode
- **Behavior**: Caches full analysis results by file hash.
- **First Run**: Performs full analysis (Normal or CSM), stores result.
- **Subsequent Runs**: Returns stored result instantly (<100ms).
- **Learning**: **Suppressed** by default. Demo runs do not pollute token statistics.
- **Use Case**: Sales demos, repeated testing of same file, instant replay.

### Using the Mode Switcher

1. Navigate to the web interface.
2. In the top-right corner, click the **Mode** dropdown (default: Normal).
3. Select **Demo Mode**.
4. Confirm the prompt.
5. Upload a chat file.
   - *First upload*: "Demo cache saved" (normal speed).
   - *Second upload*: "Demo cache hit" (instant).

### Managing Demo Cache

**Clear specific demo entry:**
```bash
python chatrel/scripts/clear_demo_cache.py --chat_hash <hash>
```

**Clear all demo entries:**
```bash
python chatrel/scripts/clear_demo_cache.py --all
```

**Configuration:**
- `DEMO_CACHE_ENABLED`: Enable/disable caching globally.
- `DEMO_POLLUTE_CSM`: Set to `True` to allow demo runs to update learning stats (default `False`).
