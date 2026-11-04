"""Batch offline model evaluation pipeline calculating AUC-PR, P@k, and calibration."""
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from pydantic import BaseModel
from src.detection.tier2_ml.supervised_classifier import SupervisedFraudClassifier


class ModelEvaluationMetrics(BaseModel):
    auc_pr: float
    precision_at_k: float
    brier_score: float
    total_test_samples: int
    fraud_samples: int


class MLOpsEvaluationPipeline:
    """Evaluates candidate model performance against golden evaluation splits."""

    @staticmethod
    def evaluate_model(
        classifier: SupervisedFraudClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int = 20,
    ) -> ModelEvaluationMetrics:
        """Calculate PR-AUC, P@k, and Brier score."""
        y_scores = classifier.model.predict_proba(X_test)[:, 1]

        # 1. AUC-PR
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auc_pr = float(round(auc(recall, precision), 4))

        # 2. Precision @ k
        top_k_indices = np.argsort(y_scores)[-k:]
        precision_at_k = float(round(np.mean(y_test[top_k_indices]), 4))

        # 3. Brier Calibration Score
        brier = float(round(brier_score_loss(y_test, y_scores), 4))

        return ModelEvaluationMetrics(
            auc_pr=auc_pr,
            precision_at_k=precision_at_k,
            brier_score=brier,
            total_test_samples=len(y_test),
            fraud_samples=int(np.sum(y_test)),
        )
