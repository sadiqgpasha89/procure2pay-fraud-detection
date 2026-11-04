"""Unit tests for Tier 2 Probabilistic Machine Learning Scoring & Graph Engine."""
import pytest
from src.core.models import RiskClassification
from src.detection.tier2_ml.calibrator import ProbabilityCalibrator
from src.detection.tier2_ml.explainability import TreeSHAPExplainer
from src.detection.tier2_ml.graph_detector import GraphCollusionDetector
from src.detection.tier2_ml.scorer import Tier2MLScorer
from src.detection.tier2_ml.supervised_classifier import SupervisedFraudClassifier
from src.detection.tier2_ml.unsupervised_detector import UnsupervisedAnomalyDetector


@pytest.mark.unit
def test_supervised_classifier_scoring():
    """Supervised classifier should output low score for clean vector, high score for fraud vector."""
    clf = SupervisedFraudClassifier()

    clean_features = {
        "po_amount_variance_ratio": 0.0,
        "vendor_bank_account_age_days": 365.0,
        "is_new_bank_account_72h": 0.0,
        "threshold_proximity_score": 0.0,
        "is_submitted_off_hours": 0.0,
        "extraction_confidence": 1.0,
        "vendor_zscore_amount_last_90d": 0.1,
        "invoice_amount_vs_90d_avg_ratio": 1.0,
        "off_hours_submission_ratio_30d": 0.0,
        "rolling_24h_spend_vs_approval_limit": 0.1,
        "smurfing_24h_pattern_detected": 0.0,
        "degree_centrality_vendor": 0.02,
        "shared_bank_account_cross_vendor_flag": 0.0,
        "shared_contact_attribute_flag": 0.0,
    }
    clean_score = clf.predict_proba(clean_features)
    assert clean_score < 0.35

    fraud_features = clean_features.copy()
    fraud_features["is_new_bank_account_72h"] = 1.0
    fraud_features["vendor_bank_account_age_days"] = 1.0
    fraud_features["invoice_amount_vs_90d_avg_ratio"] = 4.5
    fraud_features["threshold_proximity_score"] = 0.98
    fraud_features["smurfing_24h_pattern_detected"] = 1.0

    fraud_score = clf.predict_proba(fraud_features)
    assert fraud_score > clean_score


@pytest.mark.unit
def test_unsupervised_cold_start_detector():
    """Unsupervised detector should flag anomalous vector without training labels (Edge Case 1)."""
    detector = UnsupervisedAnomalyDetector()

    normal_features = {
        "po_amount_variance_ratio": 0.02,
        "vendor_bank_account_age_days": 200.0,
        "extraction_confidence": 0.98,
        "vendor_zscore_amount_last_90d": 0.0,
    }
    normal_score = detector.score_anomaly(normal_features)

    anomaly_features = {
        "po_amount_variance_ratio": 0.85,
        "vendor_bank_account_age_days": 1.0,
        "extraction_confidence": 0.40,
        "vendor_zscore_amount_last_90d": 5.0,
        "invoice_amount_vs_90d_avg_ratio": 6.0,
    }
    anomaly_score = detector.score_anomaly(anomaly_features)
    assert anomaly_score > normal_score


@pytest.mark.unit
def test_graph_collusion_detection(graph_engine):
    """Graph detector should catch shared residential address between director and approver."""
    detector = GraphCollusionDetector(graph_engine=graph_engine)
    findings = detector.get_findings(vendor_id="VEND-COLLUSIVE", approver_id="EMP-APPROVER-99")

    assert findings.entity_conflicts_detected
    assert "Shared address" in findings.details or "742 evergreen" in findings.details.lower()

    score = detector.score_network_risk(vendor_id="VEND-COLLUSIVE", approver_id="EMP-APPROVER-99")
    assert score >= 0.50


@pytest.mark.unit
def test_treeshap_explainability():
    """TreeSHAP explainer should return top-k risk drivers with positive attribution."""
    clf = SupervisedFraudClassifier()
    explainer = TreeSHAPExplainer(clf)

    features = {
        "po_amount_variance_ratio": 0.40,
        "vendor_bank_account_age_days": 1.5,
        "is_new_bank_account_72h": 1.0,
        "invoice_amount_vs_90d_avg_ratio": 4.12,
        "smurfing_24h_pattern_detected": 1.0,
    }
    contributions = explainer.explain(features, top_k=3)
    assert len(contributions) > 0
    assert contributions[0].shap_value > 0.0


@pytest.mark.unit
def test_brier_score_calibration():
    """ProbabilityCalibrator should track predictions vs outcomes and compute Brier score."""
    calibrator = ProbabilityCalibrator()
    calibrator.record_outcome(0.90, 1)
    calibrator.record_outcome(0.10, 0)
    calibrator.record_outcome(0.85, 1)

    brier = calibrator.compute_brier_score()
    assert brier < 0.10  # Well calibrated


@pytest.mark.unit
def test_tier2_composite_scorer_classification():
    """Tier 2 Scorer should categorize Low, Medium, and High risk properly."""
    scorer = Tier2MLScorer()

    low_features = {f: 0.0 for f in scorer.supervised.feature_names}
    low_features["vendor_bank_account_age_days"] = 300.0
    low_features["extraction_confidence"] = 1.0
    low_features["invoice_amount_vs_90d_avg_ratio"] = 1.0
    low_features["rolling_24h_spend_vs_approval_limit"] = 0.1

    low_result = scorer.score(low_features, vendor_id="VEND-001")
    assert low_result.classification == RiskClassification.LOW_RISK
    assert low_result.composite_score < 0.35

    high_features = low_features.copy()
    high_features["is_new_bank_account_72h"] = 1.0
    high_features["vendor_bank_account_age_days"] = 1.0
    high_features["invoice_amount_vs_90d_avg_ratio"] = 5.0
    high_features["threshold_proximity_score"] = 0.99
    high_features["smurfing_24h_pattern_detected"] = 1.0

    high_result = scorer.score(high_features, vendor_id="VEND-001")
    assert high_result.composite_score >= 0.35
    assert len(high_result.contributions) > 0
