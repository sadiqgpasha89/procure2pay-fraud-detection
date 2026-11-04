"""Statistical drift detection subsystem (Population Stability Index & Wasserstein Distance)."""
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DriftSeverity(str, Enum):
    STABLE = "STABLE"                    # PSI < 0.10
    MARGINAL_SHIFT = "MARGINAL_SHIFT"    # 0.10 <= PSI < 0.25
    SEVERE_DRIFT = "SEVERE_DRIFT"        # PSI >= 0.25


class FeatureDriftReport(BaseModel):
    """Drift evaluation report for a single feature."""
    feature_name: str
    psi: float
    severity: DriftSeverity
    wasserstein_distance: float
    alert_triggered: bool = False


class DatasetDriftReport(BaseModel):
    """Aggregate dataset drift report across all features."""
    overall_severity: DriftSeverity
    max_psi: float
    features_drifted: List[FeatureDriftReport]
    requires_shadow_retraining: bool = False
    requires_fallback_weights: bool = False


class PSICalculator:
    """Calculates Population Stability Index (PSI) between baseline and production feature distributions."""

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4
    ) -> float:
        """Compute PSI across bucketed empirical distributions.
        
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create equal-frequency quantile buckets on baseline expected data
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bucket_bounds = np.percentile(expected, percentiles)
        # Ensure distinct bounds and expand outer bounds to cover all values
        bucket_bounds = np.unique(bucket_bounds)
        if len(bucket_bounds) < 2:
            bucket_bounds = np.array([-np.inf, np.inf])
        else:
            bucket_bounds[0] = -np.inf
            bucket_bounds[-1] = np.inf

        # Calculate frequency counts in each bucket
        expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
        actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

        # Convert to relative proportions with epsilon smoothing
        expected_pct = (expected_counts / max(1, len(expected))) + epsilon
        actual_pct = (actual_counts / max(1, len(actual))) + epsilon

        # Normalize proportions to sum to 1
        expected_pct /= np.sum(expected_pct)
        actual_pct /= np.sum(actual_pct)

        # Compute PSI summation
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(round(max(0.0, psi_value), 4))


class DriftMonitor:
    """Enterprise MLOps Dual Drift Monitor (Section 8.1)."""

    def __init__(self, baseline_data: Optional[Dict[str, List[float]]] = None):
        self.baseline_data: Dict[str, np.ndarray] = {
            k: np.array(v) for k, v in (baseline_data or {}).items()
        }
        self.production_data: Dict[str, List[float]] = {}

    def set_baseline(self, feature_name: str, values: List[float]):
        """Register baseline expected distribution from training dataset."""
        self.baseline_data[feature_name] = np.array(values)

    def record_production_value(self, feature_name: str, value: float):
        """Record live feature inference value."""
        if feature_name not in self.production_data:
            self.production_data[feature_name] = []
        self.production_data[feature_name].append(float(value))

    def evaluate_feature_drift(self, feature_name: str) -> FeatureDriftReport:
        """Evaluate drift for an individual feature."""
        expected = self.baseline_data.get(feature_name, np.array([]))
        actual = np.array(self.production_data.get(feature_name, []))

        if len(expected) < 10 or len(actual) < 10:
            return FeatureDriftReport(
                feature_name=feature_name,
                psi=0.0,
                severity=DriftSeverity.STABLE,
                wasserstein_distance=0.0,
                alert_triggered=False
            )

        psi = PSICalculator.calculate_psi(expected, actual)
        w_dist = float(round(stats.wasserstein_distance(expected, actual), 4))

        if psi < 0.10:
            severity = DriftSeverity.STABLE
            alert = False
        elif 0.10 <= psi < 0.25:
            severity = DriftSeverity.MARGINAL_SHIFT
            alert = True
        else:
            severity = DriftSeverity.SEVERE_DRIFT
            alert = True

        return FeatureDriftReport(
            feature_name=feature_name,
            psi=psi,
            severity=severity,
            wasserstein_distance=w_dist,
            alert_triggered=alert
        )

    def evaluate_all(self) -> DatasetDriftReport:
        """Run complete drift evaluation across all registered features."""
        reports: List[FeatureDriftReport] = []
        max_psi = 0.0

        for f_name in self.baseline_data:
            rep = self.evaluate_feature_drift(f_name)
            reports.append(rep)
            if rep.psi > max_psi:
                max_psi = rep.psi

        if max_psi < 0.10:
            overall = DriftSeverity.STABLE
            retrain = False
            fallback = False
        elif 0.10 <= max_psi < 0.25:
            overall = DriftSeverity.MARGINAL_SHIFT
            retrain = True
            fallback = False
            logger.warn("mlops.marginal_drift_detected", max_psi=max_psi)
        else:
            overall = DriftSeverity.SEVERE_DRIFT
            retrain = True
            fallback = True
            logger.error("mlops.severe_drift_alert", max_psi=max_psi)

        return DatasetDriftReport(
            overall_severity=overall,
            max_psi=round(max_psi, 4),
            features_drifted=reports,
            requires_shadow_retraining=retrain,
            requires_fallback_weights=fallback,
        )
