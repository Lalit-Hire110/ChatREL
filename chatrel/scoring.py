"""
Scoring module for ChatREL v4
Combines sub-scores into overall health and classifies relationship type
"""

import logging
from typing import Dict, Any

from . import config

logger = logging.getLogger(__name__)


class RelationshipScorer:
    """Compute overall health score and classify relationship type."""
    
    def __init__(
        self,
        warmth_weight: float = None,
        engagement_weight: float = None,
        conflict_weight: float = None,
        stability_weight: float = None,
    ):
        """
        Initialize scorer with configurable weights.
        
        Args:
            Weights for sub-scores (default from config)
        """
        self.warmth_weight = warmth_weight or config.WARMTH_WEIGHT
        self.engagement_weight = engagement_weight or config.ENGAGEMENT_WEIGHT
        self.conflict_weight = conflict_weight or config.CONFLICT_WEIGHT
        self.stability_weight = stability_weight or config.STABILITY_WEIGHT
    
    def compute_overall_health(self, subscores: Dict[str, float]) -> float:
        """
        Compute overall health score (0-100).
        
        Formula: health = 100 * clip(
            warmth_weight * warmth
            + engagement_weight * engagement
            - conflict_weight * conflict
            + stability_weight * stability,
            0, 1
        )
        
        Args:
            subscores: Dict with warmth_score, engagement_score, conflict_score, stability_score
        
        Returns:
            Overall health score (0-100)
        """
        warmth = subscores.get("warmth_score", 0.5)
        engagement = subscores.get("engagement_score", 0.5)
        conflict = subscores.get("conflict_score", 0.0)
        stability = subscores.get("stability_score", 0.5)
        
        # Weighted combination (conflict is subtracted)
        raw_score = (
            self.warmth_weight * warmth
            + self.engagement_weight * engagement
            - self.conflict_weight * conflict
            + self.stability_weight * stability
        )
        
        # Clip to [0, 1] and scale to 0-100
        health = max(0, min(1, raw_score)) * 100
        
        return round(health, 1)
    
    def classify_relationship(
        self, 
        subscores: Dict[str, float], 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Classify relationship type using heuristic rules.
        
        Types: Couple, Crush, Friend, Family
        
        Args:
            subscores: Normalized sub-scores
            metrics: Raw metrics from aggregator
        
        Returns:
            Dict with 'type', 'confidence', 'reasoning'
        """
        warmth = subscores.get("warmth_score", 0.5)
        engagement = subscores.get("engagement_score", 0.5)
        emoji_aff = metrics.get("emoji_affinity", 0.5)
        reciprocity = metrics.get("reciprocity", 0.5)
        avg_words = metrics.get("avg_words", 0)
        emoji_density = metrics.get("emoji_density", 0)
        
        scores = {}
        
        # Couple: high warmth + high emoji affinity + balanced reciprocity
        thresholds = config.RELATIONSHIP_THRESHOLDS["couple"]
        if (warmth >= thresholds["warmth_min"]
                and emoji_aff >= thresholds["emoji_affinity_min"]
                and reciprocity >= thresholds["reciprocity_min"]):
            scores["Couple"] = 0.8 + 0.2 * warmth
        
        # Crush: high warmth + medium emoji + asymmetric
        thresholds = config.RELATIONSHIP_THRESHOLDS["crush"]
        if (warmth >= thresholds["warmth_min"]
                and emoji_aff >= thresholds["emoji_affinity_min"]
                and reciprocity < thresholds["reciprocity_max"]):
            scores["Crush"] = 0.7 + 0.3 * (1 - reciprocity)
        
        # Friend: moderate warmth + high engagement
        thresholds = config.RELATIONSHIP_THRESHOLDS["friend"]
        if (thresholds["warmth_min"] <= warmth < thresholds.get("warmth_max", 1.0)
                and engagement >= thresholds["engagement_min"]):
            scores["Friend"] = 0.6 + 0.4 * engagement
        
        # Family: moderate warmth + longer messages + low emoji density
        thresholds = config.RELATIONSHIP_THRESHOLDS["family"]
        if (thresholds["warmth_min"] <= warmth < thresholds.get("warmth_max", 1.0)
                and avg_words >= thresholds["avg_words_min"]
                and emoji_density < thresholds["emoji_density_max"]):
            scores["Family"] = 0.5 + 0.5 * (avg_words / 20)
        
        # Default to Friend if no clear match
        if not scores:
            scores["Friend"] = 0.4
        
        # Pick highest scoring type
        rel_type = max(scores, key=scores.get)
        confidence = scores[rel_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(rel_type, subscores, metrics)
        
        return {
            "type": rel_type,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "all_scores": scores,
        }
    
    def _generate_reasoning(
        self, 
        rel_type: str, 
        subscores: Dict[str, float], 
        metrics: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for classification."""
        warmth = subscores.get("warmth_score", 0.5)
        engagement = subscores.get("engagement_score", 0.5)
        
        reasons = []
        
        if rel_type == "Couple":
            reasons.append(f"High warmth ({warmth:.1%})")
            if metrics.get("emoji_affinity", 0) > 0.6:
                reasons.append("Frequent positive emojis")
            if metrics.get("reciprocity", 0) > 0.7:
                reasons.append("Balanced communication")
        
        elif rel_type == "Crush":
            reasons.append(f"High warmth ({warmth:.1%})")
            if metrics.get("reciprocity", 1) < 0.6:
                reasons.append("Asymmetric engagement (typical of crushes)")
            if metrics.get("emoji_affinity", 0) > 0.5:
                reasons.append("Romantic emoji usage")
        
        elif rel_type == "Friend":
            reasons.append(f"Good engagement ({engagement:.1%})")
            if warmth < 0.6:
                reasons.append("Moderate emotional tone")
            if metrics.get("msgs_per_day", 0) > 5:
                reasons.append("Frequent messaging")
        
        elif rel_type == "Family":
            if metrics.get("avg_words", 0) > 10:
                reasons.append("Longer messages")
            if metrics.get("emoji_density", 1) < 0.5:
                reasons.append("Low emoji usage")
            reasons.append("Balanced, practical communication")
        
        return "; ".join(reasons) if reasons else "Based on overall patterns"
    
    def generate_report(
        self, 
        subscores: Dict[str, float], 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate complete relationship report.
        
        Returns:
            Complete report with health, type, sub-scores, and metrics
        """
        overall_health = self.compute_overall_health(subscores)
        classification = self.classify_relationship(subscores, metrics)
        
        return {
            "overall_health": overall_health,
            "relationship_type": classification["type"],
            "relationship_confidence": classification["confidence"],
            "relationship_reasoning": classification["reasoning"],
            "sub_scores": {
                "warmth": round(subscores.get("warmth_score", 0.5) * 100, 1),
                "engagement": round(subscores.get("engagement_score", 0.5) * 100, 1),
                "conflict": round(subscores.get("conflict_score", 0.0) * 100, 1),
                "stability": round(subscores.get("stability_score", 0.5) * 100, 1),
            },
            "metrics": metrics,
            "scoring_weights": {
                "warmth": self.warmth_weight,
                "engagement": self.engagement_weight,
                "conflict": self.conflict_weight,
                "stability": self.stability_weight,
            },
        }


if __name__ == "__main__":
    # Test scorer
    scorer = RelationshipScorer()
    
    # Test case: Couple
    subscores = {
        "warmth_score": 0.85,
        "engagement_score": 0.75,
        "conflict_score": 0.15,
        "stability_score": 0.70,
    }
    
    metrics = {
        "emoji_affinity": 0.75,
        "reciprocity": 0.80,
        "avg_words": 12,
        "emoji_density": 1.5,
        "msgs_per_day": 15,
    }
    
    report = scorer.generate_report(subscores, metrics)
    
    print("Relationship Report:")
    print(f"  Overall Health: {report['overall_health']}/100")
    print(f"  Type: {report['relationship_type']} (confidence: {report['relationship_confidence']})")
    print(f"  Reasoning: {report['relationship_reasoning']}")
    print(f"\n  Sub-scores:")
    for k, v in report['sub_scores'].items():
        print(f"    {k}: {v}/100")
