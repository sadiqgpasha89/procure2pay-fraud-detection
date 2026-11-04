"""Local explainability layer isolating top contributing risk variables via TreeSHAP."""
from typing import Any, Dict, List, Optional
import numpy as np
import shap
import structlog
from src.core.models import ShapContribution
from src.detection.tier2_ml.supervised_classifier import SupervisedFraudClassifier

logger = structlog.get_logger(__name__)


class TreeSHAPExplainer:
    """TreeSHAP local explainer generating top-k risk driver feature attribution vectors."""

    def __init__(self, classifier: SupervisedFraudClassifier):
        self.classifier = classifier
        self.feature_names = classifier.feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP TreeExplainer on trained classifier."""
        if self.classifier.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.classifier.model)
            except Exception as e:
                logger.warn("shap.init_failed_fallback_heuristic", error=str(e))
                self.explainer = None

    def explain(self, features: Dict[str, float], top_k: int = 5) -> List[ShapContribution]:
        """Compute feature contribution attributions sorted by highest positive risk contribution."""
        vec = self.classifier.vectorize_features(features)
        contributions: List[ShapContribution] = []

        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(vec)
                # For binary classification, TreeExplainer returns array of shape (1, n_features) or list of 2 arrays
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                elif shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                else:
                    vals = shap_values[0]

                for idx, f_name in enumerate(self.feature_names):
                    val = float(vec[0, idx])
                    s_val = float(vals[idx])
                    contributions.append(
                        ShapContribution(
                            feature=f_name,
                            value=round(val, 3),
                            shap_value=round(s_val, 3)
                        )
                    )
            except Exception as e:
                logger.error("shap.calc_error_falling_back", error=str(e))
                contributions = self._heuristic_attribution(features)
        else:
            contributions = self._heuristic_attribution(features)

        # Sort by highest positive risk drivers
        contributions.sort(key=lambda x: x.shap_value, reverse=True)
        return contributions[:top_k]

    def _heuristic_attribution(self, features: Dict[str, float]) -> List[ShapContribution]:
        """Heuristic attribution fallback."""
        items = []
        for k, v in features.items():
            # Estimate risk impact based on normalized magnitude
            impact = 0.0
            if "bank" in k and v <= 3.0:
                impact = 0.381
            elif "vs_90d" in k and v > 2.0:
                impact = 0.245
            elif "smurfing" in k and v == 1.0:
                impact = 0.320
            elif "variance" in k and v > 0.10:
                impact = 0.180
            elif "proximity" in k and v > 0.5:
                impact = 0.210
            elif "conflicts" in k:
                impact = 0.280

            if impact > 0:
                items.append(ShapContribution(feature=k, value=float(v), shap_value=round(impact, 3)))
        return items
