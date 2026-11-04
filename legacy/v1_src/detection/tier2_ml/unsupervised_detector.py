"""Unsupervised anomaly detection subsystem (Isolation Forest & Reconstruction Error).

Mitigates Edge Case 1: Sparse Label Cold-Start by detecting multidimensional
anomalies without requiring labeled historical fraud cases.
"""
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import structlog
from src.detection.tier2_ml.supervised_classifier import FEATURE_ORDER

logger = structlog.get_logger(__name__)


class LinearReconstructionAutoencoder:
    """Lightweight principal-component autoencoder calculating reconstruction error L_AE = ||x - x_hat||^2."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Compute squared error between original x and reconstructed x_hat."""
        if self.mean is None or self.components is None:
            return 0.0
        x_centered = x - self.mean
        projected = np.dot(x_centered, self.components.T)
        reconstructed = np.dot(projected, self.components) + self.mean
        error = np.mean((x - reconstructed) ** 2)
        # Normal inlier error is < 1.0; error > 3.0 indicates severe reconstruction failure
        normalized = min(1.0, error / 4.0)
        return float(np.clip(normalized, 0.0, 1.0))


class UnsupervisedAnomalyDetector:
    """Ensemble of Isolation Forest and Reconstruction Autoencoder for cold-start anomaly scoring."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_ORDER
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.autoencoder = LinearReconstructionAutoencoder(n_components=4)
        self._init_baseline()

    def _generate_clean_baseline(self, n_samples: int = 300) -> np.ndarray:
        """Generate baseline normal transaction feature vectors."""
        np.random.seed(42)
        X = np.zeros((n_samples, len(self.feature_names)))
        X[:, 0] = np.random.uniform(0.0, 0.05, n_samples)          # po_amount_variance_ratio
        X[:, 1] = np.random.uniform(90.0, 700.0, n_samples)        # vendor_bank_account_age_days
        X[:, 2] = 0.0                                              # is_new_bank_account_72h
        X[:, 3] = np.random.uniform(0.0, 0.1, n_samples)           # threshold_proximity_score
        X[:, 4] = np.random.choice([0.0, 1.0], size=n_samples, p=[0.95, 0.05])  # is_submitted_off_hours
        X[:, 5] = np.random.uniform(0.92, 1.0, n_samples)          # extraction_confidence
        X[:, 6] = np.random.normal(0.0, 0.8, n_samples)            # vendor_zscore_amount_last_90d
        X[:, 7] = np.random.uniform(0.85, 1.15, n_samples)         # invoice_amount_vs_90d_avg_ratio
        X[:, 8] = np.random.uniform(0.0, 0.10, n_samples)          # off_hours_submission_ratio_30d
        X[:, 9] = np.random.uniform(0.05, 0.40, n_samples)         # rolling_24h_spend_vs_approval_limit
        X[:, 10] = 0.0                                             # smurfing_24h_pattern_detected
        X[:, 11] = np.random.uniform(0.01, 0.06, n_samples)        # degree_centrality_vendor
        X[:, 12] = 0.0                                             # shared_bank_account_cross_vendor_flag
        X[:, 13] = 0.0                                             # shared_contact_attribute_flag
        return X

    def _init_baseline(self):
        """Fit baseline normal transaction distribution."""
        X_clean = self._generate_clean_baseline(n_samples=300)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.iso_forest.fit(X_scaled)
        self.autoencoder.fit(X_scaled)

    def score_anomaly(self, features: Dict[str, float]) -> float:
        """Compute composite unsupervised anomaly score in [0.0, 1.0]."""
        vec = np.array([float(features.get(f, 0.0)) for f in self.feature_names]).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)

        # 1. Isolation Forest score: decision_function is positive for inliers (>0), negative for outliers (<0)
        iso_raw = float(self.iso_forest.decision_function(vec_scaled)[0])
        if iso_raw >= 0.0:
            # Inlier: map [0.0, 0.25] to [0.20, 0.0]
            iso_score = max(0.0, 0.20 - (iso_raw * 0.8))
        else:
            # Outlier: map [0.0, -0.3] to [0.20, 1.0]
            iso_score = min(1.0, 0.20 + abs(iso_raw) * 2.5)

        # 2. Autoencoder reconstruction error
        ae_score = self.autoencoder.reconstruction_error(vec_scaled.flatten())

        # Ensemble
        composite = (iso_score * 0.6) + (ae_score * 0.4)
        return float(np.clip(composite, 0.0, 1.0))
