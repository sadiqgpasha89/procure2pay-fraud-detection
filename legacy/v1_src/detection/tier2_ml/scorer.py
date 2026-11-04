"""Composite Tier 2 Probabilistic Machine Learning Scorer."""
import time
from typing import Any, Dict, Optional
import structlog
from src.core.config import get_settings
from src.core.models import RiskClassification, Tier2Result
from src.detection.tier2_ml.supervised_classifier import SupervisedFraudClassifier
from src.detection.tier2_ml.unsupervised_detector import UnsupervisedAnomalyDetector
from src.detection.tier2_ml.graph_detector import GraphCollusionDetector
from src.detection.tier2_ml.explainability import TreeSHAPExplainer
from src.detection.tier2_ml.calibrator import ProbabilityCalibrator

logger = structlog.get_logger(__name__)


class Tier2MLScorer:
    """Orchestrates Supervised, Unsupervised, and Graph models to compute composite fraud risk."""

    def __init__(
        self,
        supervised_classifier: Optional[SupervisedFraudClassifier] = None,
        unsupervised_detector: Optional[UnsupervisedAnomalyDetector] = None,
        graph_detector: Optional[GraphCollusionDetector] = None,
    ):
        self.settings = get_settings()
        self.supervised = supervised_classifier or SupervisedFraudClassifier()
        self.unsupervised = unsupervised_detector or UnsupervisedAnomalyDetector()
        self.graph = graph_detector or GraphCollusionDetector()
        self.explainer = TreeSHAPExplainer(self.supervised)
        self.calibrator = ProbabilityCalibrator()

        self.tau_1 = self.settings.TAU_1_LOW_RISK_THRESHOLD
        self.tau_2 = self.settings.TAU_2_HIGH_RISK_THRESHOLD

    def score(
        self,
        features: Dict[str, float],
        vendor_id: str,
        approver_id: Optional[str] = None,
        fallback_penalty: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tier2Result:
        """Compute composite risk score S in [0.0, 1.0], SHAP contributions, and classification tier."""
        start_time = time.perf_counter()

        # 1. Supervised scoring
        sup_score = self.supervised.predict_proba(features)

        # 2. Unsupervised anomaly scoring (Cold-start mitigation)
        unsup_score = self.unsupervised.score_anomaly(features)

        # 3. Graph network collusion scoring
        graph_score = self.graph.score_network_risk(vendor_id, approver_id)

        # Ensemble combination: weighted sum with dominant risk signal preservation
        weighted_sum = (sup_score * 0.50) + (unsup_score * 0.30) + (graph_score * 0.20)
        dominant_risk = max(sup_score, unsup_score * 0.85, graph_score)
        composite_raw = max(weighted_sum, dominant_risk)

        # Apply fallback penalty from degraded third-party API circuit breaker (Section 6)
        if fallback_penalty > 0:
            logger.info("tier2.applying_fallback_penalty", penalty=fallback_penalty)
            composite_raw += fallback_penalty

        # Bounded score S in [0.0, 1.0]
        composite_score = float(round(min(1.0, max(0.0, composite_raw)), 3))

        # Determine Classification Tier
        if composite_score < self.tau_1:
            classification = RiskClassification.LOW_RISK
        elif self.tau_1 <= composite_score < self.tau_2:
            classification = RiskClassification.MEDIUM_RISK
        else:
            classification = RiskClassification.HIGH_RISK

        # TreeSHAP feature attributions
        contributions = self.explainer.explain(features, top_k=5)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return Tier2Result(
            composite_score=composite_score,
            supervised_score=round(sup_score, 3),
            unsupervised_score=round(unsup_score, 3),
            graph_score=round(graph_score, 3),
            classification=classification,
            contributions=contributions,
            brier_score=self.calibrator.compute_brier_score(),
            latency_ms=round(elapsed_ms, 3),
        )
