#!/usr/bin/env python3
"""
Warm-up Prefill Script for ChatREL v4 - Contextual Sentiment Memory
Pre-populates cache and statistics from historical chat data
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatrel.parser import WhatsAppParser
from chatrel.hf_client import HFClient
from chatrel.message_processor import MessageProcessor
from chatrel.utils.nlp_cache import NLPCache
from chatrel.utils.token_stats import TokenStatsEngine, ContextSignatureGenerator
from chatrel.tasks.update_word_stats import enqueue_token_update
from chatrel import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WarmupPrefill:
    """Prefill CSM cache and statistics from historical data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.nlp_cache = NLPCache()
        self.token_engine = TokenStatsEngine()
        self.parser = WhatsAppParser()
        
        # Statistics tracking
        self.stats = {
            'messages_processed': 0,
            'messages_cached': 0,
            'tokens_learned': set(),
            'contexts_learned': set(),
            'unknown_tokens': Counter(),
            'high_variance_tokens': [],
            'unstable_contexts': [],
            'errors': 0,
            'start_time': time.time()
        }
    
    def process_chat_file(self, filepath: Path, use_hf: bool = True) -> int:
        """
        Process a single chat file and populate cache/stats.
        
        Args:
            filepath: Path to WhatsApp export
            use_hf: Whether to use HuggingFace API (required for initial learning)
        
        Returns:
            Number of messages processed
        """
        logger.info(f"Processing: {filepath}")
        
        try:
            df = self.parser.parse_file(str(filepath))
            logger.info(f"Parsed {len(df)} messages")
            
            if len(df) == 0:
                logger.warning(f"No messages in {filepath}")
                return 0
            
            # Initialize HF client if needed
            if use_hf:
                if not config.HF_TOKEN:
                    logger.error("HF_TOKEN not set - cannot run warm-up with HF API")
                    return 0
                
                hf_client = HFClient(mock_mode=False, use_cache=False)
                processor = MessageProcessor(hf_client)
                
                # Process in batches to get sentiment/toxicity
                logger.info("Running sentiment/toxicity analysis...")
                df_analyzed = processor.process_messages(df)
            else:
                # Use inference only (requires existing stats)
                logger.info("Using inference-only mode (no HF API)")
                df_analyzed = df.copy()
                
                for idx, row in df_analyzed.iterrows():
                    sent_result = self.token_engine.infer_sentiment(row['text'])
                    tox_result = self.token_engine.infer_toxicity(row['text'])
                    
                    df_analyzed.at[idx, 'sentiment'] = sent_result['score']
                    df_analyzed.at[idx, 'toxicity'] = tox_result['score']
            
            # Cache results and update token stats
            model_version = config.CSM_SENTIMENT_MODEL_VERSION
            
            for _, row in df_analyzed.iterrows():
                text = row['text']
                sentiment_score = row.get('sentiment', 0.0)
                toxicity_score = row.get('toxicity', 0.0)
                
                # Cache the message
                self.nlp_cache.set(
                    text=text,
                    model_name=config.SENTIMENT_MODEL,
                    model_version=model_version,
                    sentiment={'label': 'inferred', 'score': sentiment_score},
                    toxicity={'label': 'inferred', 'score': toxicity_score},
                    confidence=1.0 if use_hf else 0.5,
                    source='hf' if use_hf else 'inference'
                )
                self.stats['messages_cached'] += 1
                
                # Update token statistics
                tokens = ContextSignatureGenerator.tokenize(text)
                
                for i, token in enumerate(tokens):
                    self.token_engine.update_token_stats(
                        token, sentiment_score, toxicity_score
                    )
                    self.stats['tokens_learned'].add(token.lower())
                    
                    # Update context stats
                    signature = ContextSignatureGenerator.generate(tokens, i)
                    self.token_engine.update_context_stats(
                        token, signature, sentiment_score, toxicity_score
                    )
                    self.stats['contexts_learned'].add(signature)
                
                self.stats['messages_processed'] += 1
                
                if self.stats['messages_processed'] % 100 == 0:
                    logger.info(f"Processed {self.stats['messages_processed']} messages...")
            
            return len(df_analyzed)
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            self.stats['errors'] += 1
            return 0
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze token and context coverage after warm-up."""
        logger.info("Analyzing coverage...")
        
        db_path = Path(config.CSM_DB_PATH)
        
        with sqlite3.connect(db_path) as conn:
            # Token statistics
            cursor = conn.execute(
                """
                SELECT COUNT(*), AVG(sentiment_count), 
                       AVG(sentiment_variance), MAX(sentiment_variance)
                FROM word_stats
                """
            )
            token_stats = cursor.fetchone()
            
            cursor = conn.execute(
                f"""
                SELECT COUNT(*) FROM word_stats 
                WHERE sentiment_count >= {config.CSM_MIN_TOKEN_COUNT}
                """
            )
            usable_tokens = cursor.fetchone()[0]
            
            # High variance tokens
            cursor = conn.execute(
                """
                SELECT token, sentiment_variance, sentiment_count
                FROM word_stats
                WHERE sentiment_variance > ?
                ORDER BY sentiment_variance DESC
                LIMIT 50
                """,
                (config.CSM_VARIANCE_THRESHOLD,)
            )
            high_variance = [
                {'token': row[0], 'variance': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Top unknown tokens during inference
            # (This would require tracking during inference phase)
            
            # Context statistics
            cursor = conn.execute(
                "SELECT COUNT(*), AVG(sentiment_count) FROM token_context_stats"
            )
            context_stats = cursor.fetchone()
            
            cursor = conn.execute(
                f"""
                SELECT COUNT(*) FROM token_context_stats
                WHERE sentiment_count >= {config.CSM_MIN_CONTEXT_COUNT}
                """
            )
            usable_contexts = cursor.fetchone()[0]
            
            # Unstable contexts
            cursor = conn.execute(
                """
                SELECT token, context_signature, sentiment_variance, sentiment_count
                FROM token_context_stats
                WHERE is_stable = 0
                ORDER BY sentiment_variance DESC
                LIMIT 50
                """,
            )
            unstable_contexts = [
                {
                    'token': row[0],
                    'signature': row[1],
                    'variance': row[2],
                    'count': row[3]
                }
                for row in cursor.fetchall()
            ]
        
        total_tokens = token_stats[0] if token_stats else 0
        total_contexts = context_stats[0] if context_stats else 0
        
        coverage = {
            'token_coverage': {
                'total_tokens': total_tokens,
                'usable_tokens': usable_tokens,
                'coverage_percent': (usable_tokens / total_tokens * 100) if total_tokens > 0 else 0,
                'avg_count': token_stats[1] if token_stats else 0,
                'avg_variance': token_stats[2] if token_stats else 0,
                'max_variance': token_stats[3] if token_stats else 0,
            },
            'context_coverage': {
                'total_contexts': total_contexts,
                'usable_contexts': usable_contexts,
                'coverage_percent': (usable_contexts / total_contexts * 100) if total_contexts > 0 else 0,
                'avg_count': context_stats[1] if context_stats else 0,
            },
            'high_variance_tokens': high_variance,
            'unstable_contexts': unstable_contexts,
        }
        
        return coverage
    
    def estimate_hf_dependency(self, sample_messages: List[str]) -> Dict[str, Any]:
        """
        Estimate remaining HF API dependency by testing inference on sample messages.
        
        Args:
            sample_messages: List of test messages
        
        Returns:
            Dependency metrics
        """
        if not sample_messages:
            return {'hf_dependency_percent': 0, 'sample_size': 0}
        
        low_confidence_count = 0
        confidences = []
        
        for msg in sample_messages:
            result = self.token_engine.infer_sentiment(msg)
            confidences.append(result['confidence'])
            
            if result['confidence'] < config.CSM_CONFIDENCE_THRESHOLD:
                low_confidence_count += 1
        
        return {
            'sample_size': len(sample_messages),
            'low_confidence_count': low_confidence_count,
            'hf_dependency_percent': (low_confidence_count / len(sample_messages) * 100),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
        }
    
    def generate_report(self, coverage: Dict[str, Any], dependency: Dict[str, Any]) -> str:
        """Generate detailed warm-up report in Markdown format."""
        elapsed = time.time() - self.stats['start_time']
        
        report = f"""# ChatREL CSM Warm-up Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Messages Processed**: {self.stats['messages_processed']:,}
- **Messages Cached**: {self.stats['messages_cached']:,}
- **Unique Tokens Learned**: {len(self.stats['tokens_learned']):,}
- **Context Signatures**: {len(self.stats['contexts_learned']):,}
- **Errors**: {self.stats['errors']}
- **Processing Time**: {elapsed:.1f} seconds
- **Throughput**: {self.stats['messages_processed'] / elapsed:.1f} messages/sec

---

## Token Coverage

| Metric | Value |
|--------|-------|
| Total Tokens | {coverage['token_coverage']['total_tokens']:,} |
| Usable Tokens (≥{config.CSM_MIN_TOKEN_COUNT} samples) | {coverage['token_coverage']['usable_tokens']:,} |
| **Coverage %** | **{coverage['token_coverage']['coverage_percent']:.1f}%** |
| Avg Count per Token | {coverage['token_coverage']['avg_count']:.1f} |
| Avg Variance | {coverage['token_coverage']['avg_variance']:.3f} |
| Max Variance | {coverage['token_coverage']['max_variance']:.3f} |

---

## Context Coverage

| Metric | Value |
|--------|-------|
| Total Context Signatures | {coverage['context_coverage']['total_contexts']:,} |
| Usable Contexts (≥{config.CSM_MIN_CONTEXT_COUNT} samples) | {coverage['context_coverage']['usable_contexts']:,} |
| **Coverage %** | **{coverage['context_coverage']['coverage_percent']:.1f}%** |
| Avg Count per Context | {coverage['context_coverage']['avg_count']:.1f} |

---

## High Variance Tokens (Top 20)

Tokens with variance >{config.CSM_VARIANCE_THRESHOLD} (unstable for inference):

| Token | Variance | Sample Count |
|-------|----------|--------------|
"""
        for item in coverage['high_variance_tokens'][:20]:
            report += f"| `{item['token']}` | {item['variance']:.3f} | {item['count']} |\n"
        
        report += f"""
---

## Unstable Context Signatures (Top 20)

| Token | Context Signature | Variance | Count |
|-------|-------------------|----------|-------|
"""
        for item in coverage['unstable_contexts'][:20]:
            report += f"| `{item['token']}` | `{item['signature']}` | {item['variance']:.3f} | {item['count']} |\n"
        
        report += f"""
---

## HF API Dependency Estimate

Based on {dependency['sample_size']} sample messages:

| Metric | Value |
|--------|-------|
| Low Confidence Messages | {dependency['low_confidence_count']} |
| **Estimated HF Dependency** | **{dependency['hf_dependency_percent']:.1f}%** |
| Avg Inference Confidence | {dependency['avg_confidence']:.2f} |
| Min Confidence | {dependency['min_confidence']:.2f} |
| Max Confidence | {dependency['max_confidence']:.2f} |

> **Note**: With current coverage, ~{100 - dependency['hf_dependency_percent']:.1f}% of messages can be handled by inference alone.

---

## Recommendations

"""
        if coverage['token_coverage']['coverage_percent'] < 50:
            report += "- ⚠️ **Low token coverage**. Process more historical data to improve inference accuracy.\n"
        
        if dependency['hf_dependency_percent'] > 30:
            report += "- ⚠️ **High HF dependency**. Consider processing more diverse chat data.\n"
        
        if len(coverage['high_variance_tokens']) > 100:
            report += "- ⚠️ **Many unstable tokens**. Review high-variance tokens for data quality issues.\n"        
        
        if coverage['token_coverage']['coverage_percent'] >= 70 and dependency['hf_dependency_percent'] < 20:
            report += "- ✅ **Good coverage achieved**. CSM is ready for production use with minimal HF API dependency.\n"
        
        report += f"""
---

## Configuration Used

- `CSM_MIN_TOKEN_COUNT`: {config.CSM_MIN_TOKEN_COUNT}
- `CSM_MIN_CONTEXT_COUNT`: {config.CSM_MIN_CONTEXT_COUNT}
- `CSM_CONFIDENCE_THRESHOLD`: {config.CSM_CONFIDENCE_THRESHOLD}
- `CSM_VARIANCE_THRESHOLD`: {config.CSM_VARIANCE_THRESHOLD}
- `SENTIMENT_MODEL`: {config.SENTIMENT_MODEL}
- `SENTIMENT_MODEL_VERSION`: {config.CSM_SENTIMENT_MODEL_VERSION}

---

*Generated by ChatREL v4 CSM Warm-up Script*
"""
        return report
    
    def run(self, data_dir: Path, limit: Optional[int] = None, use_hf: bool = True):
        """
        Run warm-up prefill process.
        
        Args:
            data_dir: Directory containing chat files
            limit: Max number of files to process (None = all)
            use_hf: Use HuggingFace API for analysis
        """
        logger.info(f"Starting warm-up prefill from: {data_dir}")
        logger.info(f"Limit: {limit if limit else 'None'}")
        logger.info(f"Use HF API: {use_hf}")
        
        # Find all chat files
        chat_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("**/*.txt"))
        
        if limit:
            chat_files = chat_files[:limit]
        
        logger.info(f"Found {len(chat_files)} chat files")
        
        # Process each file
        for i, filepath in enumerate(chat_files, 1):
            logger.info(f"\n[{i}/{len(chat_files)}] Processing {filepath.name}")
            self.process_chat_file(filepath, use_hf=use_hf)
        
        # Analyze coverage
        logger.info("\nAnalyzing coverage...")
        coverage = self.analyze_coverage()
        
        # Estimate HF dependency (use some messages as test sample)
        logger.info("Estimating HF API dependency...")
        sample_texts = []
        for filepath in chat_files[:min(5, len(chat_files))]:
            try:
                df = self.parser.parse_file(str(filepath))
                sample_texts.extend(df['text'].sample(min(50, len(df))).tolist())
            except Exception:
                pass
        
        dependency = self.estimate_hf_dependency(sample_texts[:200])
        
        # Generate report
        logger.info("Generating report...")
        report_md = self.generate_report(coverage, dependency)
        
        # Save report
        report_path = self.output_dir / f"warmup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        logger.info(f"Report saved: {report_path}")
        
        # Save JSON data
        json_data = {
            'summary': {
                'messages_processed': self.stats['messages_processed'],
                'messages_cached': self.stats['messages_cached'],
                'unique_tokens': len(self.stats['tokens_learned']),
                'unique_contexts': len(self.stats['contexts_learned']),
                'errors': self.stats['errors'],
                'elapsed_seconds': time.time() - self.stats['start_time'],
            },
            'coverage': coverage,
            'dependency': dependency,
        }
        
        json_path = self.output_dir / f"warmup_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"JSON data saved: {json_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("WARM-UP COMPLETE")
        print("="*70)
        print(f"Messages processed: {self.stats['messages_processed']:,}")
        print(f"Tokens learned: {len(self.stats['tokens_learned']):,}")
        print(f"Token coverage: {coverage['token_coverage']['coverage_percent']:.1f}%")
        print(f"HF dependency: {dependency['hf_dependency_percent']:.1f}%")
        print(f"\nFull report: {report_path}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="ChatREL CSM Warm-up Prefill")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('sample_data'),
        help='Directory containing chat files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of files to process'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('warmup_reports'),
        help='Output directory for reports'
    )
    parser.add_argument(
        '--no-hf',
        action='store_true',
        help='Use inference only (no HuggingFace API calls)'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    warmup = WarmupPrefill(output_dir=args.output_dir)
    warmup.run(
        data_dir=args.data_dir,
        limit=args.limit,
        use_hf=not args.no_hf
    )


if __name__ == "__main__":
    main()
