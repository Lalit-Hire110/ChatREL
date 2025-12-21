-- ============================================================================
-- ChatREL v4 - Contextual Sentiment Memory (CSM) Database Schema
-- ============================================================================
-- This schema supports message-level caching, token-level statistics,
-- context-aware scoring, and decision logging for the CSM system.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. MESSAGE CACHE
-- ----------------------------------------------------------------------------
-- Stores exact message → sentiment/toxicity mappings with version tracking
-- Enables instant score reuse for previously analyzed messages
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS message_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT NOT NULL,
    text_sanitized TEXT,  -- NULL when CSM_PRIVACY_MODE=True
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    
    -- Sentiment analysis results
    sentiment_label TEXT,
    sentiment_score REAL,
    
    -- Toxicity analysis results
    toxicity_label TEXT,
    toxicity_score REAL,
    
    -- Metadata
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'hf',  -- 'hf' | 'inference' | 'hybrid'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    
    UNIQUE(text_hash, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_message_cache_hash 
    ON message_cache(text_hash);

CREATE INDEX IF NOT EXISTS idx_message_cache_created 
    ON message_cache(created_at);

CREATE INDEX IF NOT EXISTS idx_message_cache_model 
    ON message_cache(model_name, model_version);


-- ----------------------------------------------------------------------------
-- 2. WORD STATISTICS (Token-level aggregates)
-- ----------------------------------------------------------------------------
-- Maintains global statistics for each token across all messages
-- Uses Welford's algorithm for incremental mean and variance calculation
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS word_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT NOT NULL UNIQUE,
    
    -- Sentiment statistics (Welford algorithm)
    sentiment_mean REAL DEFAULT 0.0,
    sentiment_m2 REAL DEFAULT 0.0,  -- For variance calculation
    sentiment_count INTEGER DEFAULT 0,
    sentiment_variance REAL DEFAULT 0.0,  -- Computed from m2
    
    -- Toxicity statistics (Welford algorithm)
    toxicity_mean REAL DEFAULT 0.0,
    toxicity_m2 REAL DEFAULT 0.0,
    toxicity_count INTEGER DEFAULT 0,
    toxicity_variance REAL DEFAULT 0.0,
    
    -- Emoji boost (from lexicon, static)
    emoji_boost REAL DEFAULT 0.0,
    
    -- Stability flags
    is_stable BOOLEAN DEFAULT 1,  -- 0 if variance exceeds threshold
    
    -- Metadata
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_word_stats_token 
    ON word_stats(token);

CREATE INDEX IF NOT EXISTS idx_word_stats_count 
    ON word_stats(sentiment_count DESC);

CREATE INDEX IF NOT EXISTS idx_word_stats_stable 
    ON word_stats(is_stable);


-- ----------------------------------------------------------------------------
-- 3. TOKEN CONTEXT STATISTICS
-- ----------------------------------------------------------------------------
-- Tracks sentiment/toxicity for tokens in specific contextual patterns
-- Context signatures include negation, capitalization, emoji presence, etc.
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS token_context_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT NOT NULL,
    context_signature TEXT NOT NULL,
    
    -- Sentiment statistics
    sentiment_mean REAL DEFAULT 0.0,
    sentiment_m2 REAL DEFAULT 0.0,
    sentiment_count INTEGER DEFAULT 0,
    sentiment_variance REAL DEFAULT 0.0,
    
    -- Toxicity statistics
    toxicity_mean REAL DEFAULT 0.0,
    toxicity_m2 REAL DEFAULT 0.0,
    toxicity_count INTEGER DEFAULT 0,
    toxicity_variance REAL DEFAULT 0.0,
    
    -- Stability flags
    is_stable BOOLEAN DEFAULT 1,
    
    -- Metadata
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(token, context_signature)
);

CREATE INDEX IF NOT EXISTS idx_context_token 
    ON token_context_stats(token);

CREATE INDEX IF NOT EXISTS idx_context_signature 
    ON token_context_stats(context_signature);

CREATE INDEX IF NOT EXISTS idx_context_count 
    ON token_context_stats(sentiment_count DESC);

CREATE INDEX IF NOT EXISTS idx_context_stable 
    ON token_context_stats(is_stable);


-- ----------------------------------------------------------------------------
-- 4. DECISION LOG (Debug mode only)
-- ----------------------------------------------------------------------------
-- Stores decision trace for each message analysis when CSM_DEBUG_DECISIONS=True
-- Useful for debugging, performance tuning, and confidence analysis
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_hash TEXT NOT NULL,
    
    -- Decision details
    resolution_source TEXT NOT NULL,  -- 'cache' | 'inference' | 'hybrid' | 'hf'
    confidence_score REAL,
    variance_factor REAL,
    context_matches_count INTEGER DEFAULT 0,
    token_matches_count INTEGER DEFAULT 0,
    unknown_tokens_count INTEGER DEFAULT 0,
    
    -- Decision reason
    decision_reason TEXT,
    
    -- Performance metrics
    lookup_time_ms REAL,
    inference_time_ms REAL,
    hf_api_time_ms REAL,
    total_time_ms REAL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decision_log_created 
    ON decision_log(created_at);

CREATE INDEX IF NOT EXISTS idx_decision_log_source 
    ON decision_log(resolution_source);

CREATE INDEX IF NOT EXISTS idx_decision_log_confidence 
    ON decision_log(confidence_score);


-- ----------------------------------------------------------------------------
-- 5. SYNC STATUS (Redis ↔ DB synchronization tracking)
-- ----------------------------------------------------------------------------
-- Tracks the last successful sync between Redis and database
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sync_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Single row table
    last_sync_at TIMESTAMP,
    tokens_synced INTEGER DEFAULT 0,
    contexts_synced INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    last_error TEXT
);

-- Insert initial row
INSERT OR IGNORE INTO sync_status (id, last_sync_at) 
VALUES (1, CURRENT_TIMESTAMP);


-- ----------------------------------------------------------------------------
-- 6. METADATA TABLE
-- ----------------------------------------------------------------------------
-- Stores schema version and system metadata
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS csm_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO csm_metadata (key, value) VALUES 
    ('schema_version', '1.0'),
    ('created_at', CURRENT_TIMESTAMP),
    ('csm_enabled', 'true');


-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- Top tokens by usage count
CREATE VIEW IF NOT EXISTS v_top_tokens AS
SELECT 
    token,
    sentiment_count,
    sentiment_mean,
    sentiment_variance,
    is_stable
FROM word_stats
ORDER BY sentiment_count DESC
LIMIT 100;

-- Unstable tokens (high variance)
CREATE VIEW IF NOT EXISTS v_unstable_tokens AS
SELECT 
    token,
    sentiment_count,
    sentiment_variance,
    toxicity_variance
FROM word_stats
WHERE is_stable = 0
ORDER BY sentiment_variance DESC;

-- Cache statistics
CREATE VIEW IF NOT EXISTS v_cache_stats AS
SELECT 
    source,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(access_count) as avg_access_count
FROM message_cache
GROUP BY source;

-- Decision log summary (when debug mode enabled)
CREATE VIEW IF NOT EXISTS v_decision_summary AS
SELECT 
    resolution_source,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    AVG(total_time_ms) as avg_latency_ms,
    AVG(context_matches_count) as avg_context_matches
FROM decision_log
GROUP BY resolution_source;


-- ============================================================================
-- TRIGGERS FOR AUTOMATIC MAINTENANCE
-- ============================================================================

-- Update last_accessed_at on cache hit
CREATE TRIGGER IF NOT EXISTS trg_update_cache_access
AFTER UPDATE ON message_cache
FOR EACH ROW
WHEN NEW.access_count > OLD.access_count
BEGIN
    UPDATE message_cache 
    SET last_accessed_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Update is_stable flag when variance changes
CREATE TRIGGER IF NOT EXISTS trg_update_word_stability
AFTER UPDATE OF sentiment_variance ON word_stats
FOR EACH ROW
BEGIN
    UPDATE word_stats
    SET is_stable = CASE 
        WHEN NEW.sentiment_variance > 0.35 THEN 0  -- CSM_VARIANCE_THRESHOLD
        ELSE 1
    END
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_update_context_stability
AFTER UPDATE OF sentiment_variance ON token_context_stats
FOR EACH ROW
BEGIN
    UPDATE token_context_stats
    SET is_stable = CASE 
        WHEN NEW.sentiment_variance > 0.35 THEN 0
        ELSE 1
    END
    WHERE id = NEW.id;
END;


-- ============================================================================
-- CLEANUP QUERIES (Run periodically)
-- ============================================================================

-- Delete old decision logs (>30 days)
-- DELETE FROM decision_log WHERE created_at < datetime('now', '-30 days');

-- Vacuum database (reclaim space)
-- VACUUM;

-- Analyze tables for query optimization
-- ANALYZE;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
