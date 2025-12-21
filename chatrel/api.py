"""
Flask API for ChatREL v4
Web interface for chat analysis
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import io
import base64

from flask import Flask, request, render_template, jsonify, redirect, url_for
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .parser import WhatsAppParser
from .hf_client import HFClient
from .message_processor import MessageProcessor
from .aggregator import Aggregator
from .scoring import RelationshipScorer
from .privacy import pseudonymize_dataframe
from .utils.redis_client import get_redis_client
from .tasks.celery_app import celery as celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=str(config.PROJECT_ROOT / "web" / "templates"),
    static_folder=str(config.PROJECT_ROOT / "web" / "static"),
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = config.PROJECT_ROOT / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize components
hf_client = None
parser = WhatsAppParser()
aggregator = Aggregator()
scorer = RelationshipScorer()


def get_hf_client():
    """Lazy-load HF client."""
    global hf_client
    if hf_client is None:
        # Check if mock mode (for demo without API key)
        mock_mode = not config.HF_TOKEN or config.HF_TOKEN == "your_huggingface_token_here"
        hf_client = HFClient(mock_mode=mock_mode)
        if mock_mode:
            logger.warning("Running in MOCK mode - using simulated API responses")
    return hf_client


@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload or pasted text."""
    try:
        # Get anonymization preference and mode
        anonymize = request.form.get('anonymize') == 'on'
        mode = request.form.get('mode', config.DEFAULT_RUN_MODE)
        
        # Check if file or text provided
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            # Validate file size (5MB limit)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to start
            
            if file_size > 5 * 1024 * 1024:
                return render_template('index.html', error="File too large! Maximum size is 5MB.")
            
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = app.config['UPLOAD_FOLDER'] / filename
            file.save(filepath)
        
        elif 'chat_text' in request.form and request.form['chat_text'].strip():
            text = request.form['chat_text']
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pasted.txt"
            filepath = app.config['UPLOAD_FOLDER'] / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
        
        else:
            return render_template('index.html', error="Please provide a file or paste text")
        
        # Analyze the file with anonymization setting
        return redirect(url_for('analyze', filepath=str(filepath), anonymize=str(anonymize), mode=mode))
    
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return render_template('index.html', error=str(e))


@app.route('/analyze')
def analyze():
    """Analyze uploaded chat and show results."""
    filepath = request.args.get('filepath')
    anonymize = request.args.get('anonymize', 'False') == 'True'
    mode = request.args.get('mode', config.DEFAULT_RUN_MODE)
    
    if not filepath or not Path(filepath).exists():
        return redirect(url_for('index'))
    
    try:
        # Use new analysis engine with report generation
        from .analysis_engine import generate_report
        
        report = generate_report(
            filepath=Path(filepath),
            use_nlp=config.USE_NLP,  # Use config default
            anonymize=anonymize,
            use_cache=True,
            mode=mode,
        )
        
        # DEBUG: Log report structure
        logger.info(f"Report generated: mode={report['mode']}, "
                   f"health={report['scores']['overall_health']['normalized']:.1f}")
        
        return render_template('report.html', report=report)
    
    except ValueError as e:
        # Configuration errors (missing HF token, etc.)
        logger.error(f"Configuration error: {e}")
        return render_template('index.html', error=f"Configuration error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return render_template('index.html', error=f"Analysis failed: {str(e)}")


@app.route('/api/query', methods=['POST'])
def api_query():
    """Answer questions about the report."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get report from session or generate fresh
        # For now, we'll expect the client to send the report in the request
        # In production, you might cache it in session or Redis
        report = data.get('report')
        
        if not report:
            return jsonify({"error": "No report data provided"}), 400
        
        # Answer the query
        from .query_engine import answer_query
        result = answer_query(report, question)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({
            "answer": "I encountered an error processing your question. Please try rephrasing it.",
            "intent": "ERROR",
            "target": None,
            "provenance": [],
            "confidence": 0.0,
            "error": str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """JSON API endpoint for analysis."""
    try:
        data = request.get_json()
        
        if 'text' in data:
            # Analyze raw text
            df = parser.parse_text(data['text'])
        elif 'messages' in data:
            # Analyze pre-structured messages
            df = pd.DataFrame(data['messages'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            return jsonify({"error": "Provide 'text' or 'messages'"}), 400
        
        # Process
        client = get_hf_client()
        processor = MessageProcessor(client)
        df = processor.process_messages(df)
        
        # Aggregate
        window = aggregator.create_message_window(df)
        metrics = aggregator.compute_metrics(window)
        subscores = aggregator.compute_subscores(metrics)
        
        # Score
        report = scorer.generate_report(subscores, metrics)
        
        return jsonify(report)
    
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint for Docker and monitoring."""
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'csm_enabled': config.CSM_ENABLED,
        'components': {}
    }
    
    # Check Redis
    redis_status = {'status': 'down', 'latency_ms': None}
    try:
        client = get_redis_client(max_retries=0)
        if client:
            start = datetime.now()
            client.ping()
            latency = (datetime.now() - start).total_seconds() * 1000
            redis_status = {'status': 'up', 'latency_ms': round(latency, 2)}
    except Exception as e:
        redis_status['error'] = str(e)
    status['components']['redis'] = redis_status
    
    # Check Celery
    celery_status = {'status': 'unknown'}
    if config.CSM_ENABLED:
        try:
            # Simple check: can we inspect?
            # Note: this might be slow, so use short timeout if possible
            # For a quick health check, maybe just checking connection is enough
            with celery_app.connection_or_acquire() as conn:
                celery_status = {'status': 'connected' if conn.connected else 'disconnected'}
        except Exception as e:
            celery_status = {'status': 'down', 'error': str(e)}
    status['components']['celery'] = celery_status
    
    # Check DB
    db_status = {'status': 'down'}
    try:
        import sqlite3
        with sqlite3.connect(config.CSM_DB_PATH) as conn:
            conn.execute("SELECT 1")
        db_status = {'status': 'up'}
    except Exception as e:
        db_status['error'] = str(e)
    status['components']['db'] = db_status
    
    # Determine overall health
    if redis_status['status'] == 'down' or db_status['status'] == 'down':
        status['status'] = 'degraded'
        return jsonify(status), 503
        
    return jsonify(status)


def generate_sentiment_plot(df: pd.DataFrame) -> str:
    """Generate sentiment over time plot as base64 PNG."""
    if len(df) == 0:
        return ""
    
    # Prepare data
    plot_df = df[df["combined_sentiment"].notna()].copy()
    if len(plot_df) == 0:
        return ""
    
    plot_df = plot_df.sort_values("timestamp")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot sentiment over time
    ax.plot(
        plot_df["timestamp"],
        plot_df["combined_sentiment"],
        marker='o',
        markersize=3,
        linestyle='-',
        linewidth=1,
        alpha=0.7,
    )
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Labels
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment")
    ax.set_title("Sentiment Over Time")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_base64}"


def get_top_messages(df: pd.DataFrame, n: int = 20) -> list:
    """Get top N messages sorted by interesting features."""
    if len(df) == 0:
        return []
    
    # Filter valid messages
    valid = df[~df["is_media"] & ~df["is_system"] & df["combined_sentiment"].notna()].copy()
    
    if len(valid) == 0:
        return []
    
    # Sort by absolute sentiment + toxicity (most interesting first)
    valid["interest_score"] = (
        abs(valid["combined_sentiment"]) * 2 
        + valid["toxicity_score"] 
        + valid["emoji_count"] * 0.1
    )
    
    top = valid.nlargest(min(n, len(valid)), "interest_score")
    
    messages = []
    for _, row in top.iterrows():
        messages.append({
            "sender": row["sender"],
            "text": row["text"][:200],  # Truncate long messages
            "sentiment": round(row["combined_sentiment"], 2),
            "toxicity": round(row["toxicity_score"], 2),
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
        })
    
    return messages


if __name__ == '__main__':
    # Validate config
    valid, msg = config.validate_config()
    if not valid:
        logger.warning(f"Config validation: {msg}")
    
    # Run Flask app
    logger.info(f"Starting server (debug=True, use_reloader={config.DEV_USE_RELOADER})")
    app.run(debug=True, use_reloader=config.DEV_USE_RELOADER, host='0.0.0.0', port=5000)
