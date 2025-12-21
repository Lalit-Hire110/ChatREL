-- Migration: Add demo_chat_cache table
-- Created: 2025-11-30

CREATE TABLE IF NOT EXISTS demo_chat_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_hash TEXT NOT NULL,
    demo_profile TEXT NOT NULL DEFAULT 'demo_v1',
    mode TEXT NOT NULL, -- 'demo' kept for clarity
    result_json TEXT NOT NULL, -- serialized analysis result (JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    UNIQUE(chat_hash, demo_profile)
);

CREATE INDEX IF NOT EXISTS idx_demo_chat_hash ON demo_chat_cache(chat_hash);
