import pytest
import os
import sqlite3
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from chatrel import config
from chatrel.utils import demo_cache
from chatrel.analysis_engine import run_analysis

# Mock data
SAMPLE_CHAT = b"25/12/23, 14:30 - Alice: Hello\n25/12/23, 14:31 - Bob: Hi there"
SAMPLE_HASH = demo_cache.compute_chat_hash(SAMPLE_CHAT)

@pytest.fixture
def db_connection(tmp_path):
    """Setup test database."""
    original_db_path = config.CSM_DB_PATH
    test_db_path = str(tmp_path / "test_csm.db")
    config.CSM_DB_PATH = test_db_path
    
    # Create table
    conn = sqlite3.connect(test_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS demo_chat_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_hash TEXT NOT NULL,
            demo_profile TEXT NOT NULL DEFAULT 'demo_v1',
            mode TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            UNIQUE(chat_hash, demo_profile)
        )
    """)
    conn.commit()
    conn.close()
    
    yield
    
    # Cleanup
    config.CSM_DB_PATH = original_db_path
    # No manual removal needed as tmp_path is handled by pytest

def test_compute_hash():
    h1 = demo_cache.compute_chat_hash(b"test")
    h2 = demo_cache.compute_chat_hash(b"test")
    h3 = demo_cache.compute_chat_hash(b"other")
    assert h1 == h2
    assert h1 != h3

def test_demo_cache_flow(db_connection):
    """Test full flow: upsert -> get -> hit"""
    
    # 1. Ensure empty initially
    assert demo_cache.get_demo_result(SAMPLE_HASH) is None
    
    # 2. Upsert result
    result = {"mode": "demo", "scores": {"health": 80}}
    demo_cache.upsert_demo_result(SAMPLE_HASH, result)
    
    # 3. Get result
    cached = demo_cache.get_demo_result(SAMPLE_HASH)
    assert cached is not None
    assert cached["scores"]["health"] == 80
    
    # 4. Verify access count updated (requires checking DB directly)
    conn = sqlite3.connect(config.CSM_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT access_count FROM demo_chat_cache WHERE chat_hash=?", (SAMPLE_HASH,))
    count = cursor.fetchone()[0]
    conn.close()
    # 1 from insert (default 1) + 1 from get = 2? 
    # Wait, upsert sets default 1. get increments.
    # So after upsert it is 1. After get it should be 2.
    # But get_demo_result does update access_count.
    # Let's check logic: upsert sets default 1. get increments.
    # So count should be 2.
    assert count == 2

@patch("chatrel.analysis_engine.WhatsAppParser")
@patch("chatrel.analysis_engine.extract_structural_features")
@patch("chatrel.analysis_engine.calculate_engagement")
@patch("chatrel.analysis_engine.calculate_warmth")
@patch("chatrel.analysis_engine.calculate_conflict")
@patch("chatrel.analysis_engine.calculate_stability")
@patch("chatrel.analysis_engine.calculate_overall_health")
@patch("chatrel.analysis_engine.predict_relationship_type")
def test_run_analysis_demo_mode(
    mock_predict, mock_health, mock_stab, mock_conf, mock_warm, mock_eng, mock_struct, mock_parser, db_connection, tmp_path
):
    """Test run_analysis with demo mode."""
    
    # Setup mocks
    mock_parser.return_value.parse_file.return_value = pd.DataFrame({"sender": ["A"], "text": ["Hi"], "timestamp": [pd.Timestamp.now()]})
    mock_struct.return_value = {"global": {"total_messages": 1, "days_active": 1, "date_range_start": "now", "date_range_end": "now"}, "per_sender": {}}
    mock_eng.return_value = {"normalized": 50}
    mock_warm.return_value = {"normalized": 50}
    mock_conf.return_value = {"normalized": 50}
    mock_stab.return_value = {"normalized": 50}
    mock_health.return_value = {"normalized": 50, "confidence": 1.0}
    mock_predict.return_value = {"type": "friend", "confidence": 1.0}
    
    # Create dummy file
    chat_file = tmp_path / "chat.txt"
    chat_file.write_bytes(SAMPLE_CHAT)
    
    # 1. Run first time (Miss)
    result1 = run_analysis(chat_file, use_nlp=False, mode="demo")
    assert result1["mode"] == "formula_only" # mode_str inside result is based on NLP setting
    
    # Verify it was cached
    cached = demo_cache.get_demo_result(SAMPLE_HASH)
    assert cached is not None
    
    # 2. Run second time (Hit)
    # Reset mocks to ensure they are NOT called
    mock_parser.reset_mock()
    
    result2 = run_analysis(chat_file, use_nlp=False, mode="demo")
    
    # Should be identical
    assert result2["scores"]["overall_health"]["normalized"] == 50
    
    # Parser should NOT be called (because we return early)
    # Wait, run_analysis calls parser? 
    # Logic:
    # if mode == 'demo': check cache. If hit, return.
    # So parser should not be called.
    # However, run_analysis reads file bytes to compute hash.
    # But it does NOT call parser.parse_file if cache hit.
    mock_parser.return_value.parse_file.assert_not_called()

