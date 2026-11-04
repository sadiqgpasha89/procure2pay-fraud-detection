"""Supervised Machine Learning Classifier for P2P fraud pattern detection."""
import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog
from sklearn.ensemble import GradientBoostingClassifier

logger = structlog.get_logger(__name__)

FEATURE_ORDER = [
    "po_amount_variance_ratio",
    "vendor_bank_account_age_days",
    "is_new_bank_account_72h",
    "threshold_proximity_score",
    "is_submitted_off_hours",
    "extraction_confidence",
    "vendor_zscore_amount_last_90d",
    "invoice_amount_vs_90d_avg_ratio",
    "off_hours_submission_ratio_30d",
    "rolling_24h_spend_vs_approval_limit",
    "smurfing_24h_pattern_detected",
    "degree_centrality_vendor",
    "shared_bank_account_cross_vendor_flag",
    "shared_contact_attribute_flag",
]


class SupervisedFraudClassifier:
    """Supervised Gradient Boosted Classifier for operational fraud probability scoring."""

    def __init__(self, model_path: Optional[str] = None):
        self.feature_names = FEATURE_ORDER
        self.model: Optional[GradientBoostingClassifier] = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._init_default_trained_model()

    def _init_default_trained_model(self):
        """Train a baseline reference model with synthesized calibrated seed distributions."""
        X_clean = np.array([
            [0.0, 300.0, 0.0, 0.0, 0.0, 0.98, 0.1, 1.0, 0.05, 0.2, 0.0, 0.05, 0.0, 0.0],
            [0.02, 180.0, 0.0, 0.0, 0.0, 0.95, 0.2, 1.05, 0.0, 0.1, 0.0, 0.02, 0.0, 0.0],
            [0.05, 500.0, 0.0, 0.0, 0.0, 0.99, -0.3, 0.9, 0.0, 0.3, 0.0, 0.01, 0.0, 0.0],
            [0.01, 90.0, 0.0, 0.1, 0.0, 0.92, 0.5, 1.1, 0.1, 0.4, 0.0, 0.04, 0.0, 0.0],
            [0.04, 400.0, 0.0, 0.0, 0.0, 0.96, 0.0, 1.0, 0.0, 0.15, 0.0, 0.03, 0.0, 0.0],
        ])
        y_clean = np.zeros(len(X_clean))

        X_fraud = np.array([
            # Bank changed <72h, near threshold split
            [0.25, 1.5, 1.0, 0.98, 1.0, 0.90, 4.12, 4.5, 0.8, 0.99, 1.0, 0.35, 1.0, 1.0],
            # Phantom delivery, massive PO variance
            [0.85, 20.0, 0.0, 0.0, 1.0, 0.60, 5.5, 6.0, 0.9, 0.2, 0.0, 0.40, 0.0, 1.0],
            # Shared bank account, shell company
            [0.10, 5.0, 1.0, 0.70, 0.0, 0.80, 2.5, 3.2, 0.5, 0.8, 1.0, 0.60, 1.0, 1.0],
            # Off-contract smurfing
            [0.0, 60.0, 0.0, 0.95, 1.0, 0.92, 3.8, 4.0, 0.7, 1.2, 1.0, 0.20, 0.0, 0.0],
            # Collusive approver
            [0.15, 200.0, 0.0, 0.85, 0.0, 0.95, 3.0, 3.5, 0.4, 0.9, 0.0, 0.50, 0.0, 1.0],
        ])
        y_fraud = np.ones(len(X_fraud))

        X = np.vstack([X_clean, X_fraud])
        y = np.concatenate([y_clean, y_fraud])

        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.model.fit(X, y)
        logger.info("supervised_classifier.initialized_default_model")

    def vectorize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Map feature dictionary into ordered numpy vector."""
        vec = [float(features.get(f_name, 0.0)) for f_name in self.feature_names]
        return np.array(vec).reshape(1, -1)

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Compute fraud probability score P(Fraud | x) in [0.0, 1.0]."""
        if self.model is None:
            return 0.5
        vec = self.vectorize_features(features)
        proba = self.model.predict_proba(vec)[0, 1]
        return float(proba)

    def save_model(self, path: str):
        """Persist model artifact."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load persisted model artifact."""
        self.model = joblib.load(path)
