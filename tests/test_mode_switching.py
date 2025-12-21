import pytest
from unittest.mock import patch, MagicMock
from chatrel import config
from chatrel.analysis_engine import run_analysis
import chatrel.csm_processor # Ensure module is loaded for patching
import pandas as pd

@patch("chatrel.analysis_engine.WhatsAppParser")
@patch("chatrel.analysis_engine.extract_structural_features")
@patch("chatrel.analysis_engine.calculate_engagement")
@patch("chatrel.analysis_engine.calculate_warmth")
@patch("chatrel.analysis_engine.calculate_conflict")
@patch("chatrel.analysis_engine.calculate_stability")
@patch("chatrel.analysis_engine.calculate_overall_health")
@patch("chatrel.analysis_engine.predict_relationship_type")
def test_csm_mode_enables_csm(
    mock_predict, mock_health, mock_stab, mock_conf, mock_warm, mock_eng, mock_struct, mock_parser, tmp_path
):
    """Verify that mode='csm' forces CSM_ENABLED=True."""
    
    # Setup mocks
    mock_parser.return_value.parse_file.return_value = pd.DataFrame({"sender": ["A"], "text": ["Hi"], "timestamp": [pd.Timestamp.now()], "is_media": [False], "is_system": [False]})
    mock_struct.return_value = {"global": {"total_messages": 1, "days_active": 1, "date_range_start": "now", "date_range_end": "now"}, "per_sender": {}}
    mock_health.return_value = {"normalized": 50, "confidence": 1.0}
    mock_predict.return_value = {"type": "friend", "confidence": 1.0}
    
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text("Hi")
    
    # Mock CSMMessageProcessor to verify it's initialized
    with patch("chatrel.csm_processor.CSMMessageProcessor") as MockCSM:
        MockCSM.return_value.process_messages.return_value = pd.DataFrame()
        
        # 1. Run with mode='csm'
        # Ensure config.CSM_ENABLED is False initially to prove override works
        original_csm = config.CSM_ENABLED
        config.CSM_ENABLED = False
        
        try:
            run_analysis(chat_file, use_nlp=True, mode="csm")
            
            # Should have initialized CSMMessageProcessor
            MockCSM.assert_called()
            
        finally:
            config.CSM_ENABLED = original_csm

def test_demo_mode_suppresses_learning(tmp_path):
    """Verify that mode='demo' sets suppress_learning=True."""
    
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text("Hi")
    
    with patch("chatrel.analysis_engine.WhatsAppParser") as mock_parser, \
         patch("chatrel.analysis_engine.extract_structural_features"), \
         patch("chatrel.analysis_engine.calculate_overall_health"), \
         patch("chatrel.analysis_engine.predict_relationship_type"), \
         patch("chatrel.csm_processor.CSMMessageProcessor") as MockCSM:
        
        mock_parser.return_value.parse_file.return_value = pd.DataFrame({"sender": ["A"], "text": ["Hi"], "timestamp": [pd.Timestamp.now()], "is_media": [False], "is_system": [False]})
        
        # Run with mode='demo' and use_nlp=True (to trigger CSM logic)
        # Assuming CSM_ENABLED is True or forced True? 
        # Wait, run_analysis logic:
        # use_csm = config.CSM_ENABLED
        # if mode == 'csm': use_csm = True
        # if use_csm: ...
        
        # So for demo mode to use CSM processor, config.CSM_ENABLED must be True.
        original_csm = config.CSM_ENABLED
        config.CSM_ENABLED = True
        
        try:
            run_analysis(chat_file, use_nlp=True, mode="demo")
            
            # Verify suppress_learning=True was passed
            call_args = MockCSM.call_args
            assert call_args.kwargs.get('suppress_learning') is True
            
        finally:
            config.CSM_ENABLED = original_csm

