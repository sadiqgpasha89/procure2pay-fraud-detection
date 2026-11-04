"""Probability calibration and Brier score tracking."""
from typing import List, Tuple
import numpy as np


class ProbabilityCalibrator:
    """Evaluates probability calibration and Brier calibration score."""

    def __init__(self):
        self.predictions: List[float] = []
        self.outcomes: List[int] = []

    def record_outcome(self, predicted_prob: float, true_outcome: int):
        """Record prediction and subsequent delayed ground-truth label (1 for fraud, 0 for clean)."""
        self.predictions.append(float(predicted_prob))
        self.outcomes.append(int(true_outcome))

    def compute_brier_score(self) -> float:
        """Calculate Brier calibration score: (1/N) * sum((f_t - o_t)^2).
        
        A lower score (closer to 0) indicates well-calibrated probabilities.
        """
        if not self.predictions:
            return 0.0
        preds = np.array(self.predictions)
        outs = np.array(self.outcomes)
        brier = np.mean((preds - outs) ** 2)
        return float(round(brier, 4))

    def calibrate(self, raw_score: float) -> float:
        """Apply Platt scaling / sigmoid calibration."""
        # Standard Platt sigmoid mapping
        calibrated = 1.0 / (1.0 + np.exp(-1.5 * (raw_score - 0.5) * 4.0))
        return float(np.clip(calibrated, 0.0, 1.0))
