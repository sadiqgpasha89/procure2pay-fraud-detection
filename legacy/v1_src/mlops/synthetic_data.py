"""Synthetic fraud scenario and feature vector generator for cold-start seeding & evaluations."""
from typing import Dict, List, Tuple
import numpy as np
from src.detection.tier2_ml.supervised_classifier import FEATURE_ORDER


class SyntheticFraudGenerator:
    """Generates parameterized synthetic fraud and legitimate transaction vectors."""

    @staticmethod
    def generate_clean_batch(n_samples: int = 100) -> np.ndarray:
        """Generate legitimate, normal transaction feature vectors."""
        np.random.seed(42)
        X = np.zeros((n_samples, len(FEATURE_ORDER)))

        # Feature order:
        # 0: po_amount_variance_ratio (< 0.05)
        # 1: vendor_bank_account_age_days (90 to 700 days)
        # 2: is_new_bank_account_72h (0.0)
        # 3: threshold_proximity_score (0.0 to 0.1)
        # 4: is_submitted_off_hours (0.05 probability)
        # 5: extraction_confidence (0.90 to 1.0)
        # 6: vendor_zscore_amount_last_90d (-1.0 to 1.5)
        # 7: invoice_amount_vs_90d_avg_ratio (0.8 to 1.2)
        # 8: off_hours_submission_ratio_30d (0.0 to 0.1)
        # 9: rolling_24h_spend_vs_approval_limit (0.1 to 0.5)
        # 10: smurfing_24h_pattern_detected (0.0)
        # 11: degree_centrality_vendor (0.01 to 0.08)
        # 12: shared_bank_account_cross_vendor_flag (0.0)
        # 13: shared_contact_attribute_flag (0.0)

        X[:, 0] = np.random.uniform(0.0, 0.05, n_samples)
        X[:, 1] = np.random.uniform(90.0, 700.0, n_samples)
        X[:, 2] = 0.0
        X[:, 3] = np.random.uniform(0.0, 0.1, n_samples)
        X[:, 4] = np.random.choice([0.0, 1.0], size=n_samples, p=[0.95, 0.05])
        X[:, 5] = np.random.uniform(0.92, 1.0, n_samples)
        X[:, 6] = np.random.normal(0.0, 0.8, n_samples)
        X[:, 7] = np.random.uniform(0.85, 1.15, n_samples)
        X[:, 8] = np.random.uniform(0.0, 0.10, n_samples)
        X[:, 9] = np.random.uniform(0.05, 0.40, n_samples)
        X[:, 10] = 0.0
        X[:, 11] = np.random.uniform(0.01, 0.06, n_samples)
        X[:, 12] = 0.0
        X[:, 13] = 0.0

        return X

    @staticmethod
    def generate_fraud_scenario(scenario_type: str) -> Dict[str, float]:
        """Generate a single feature vector representing a specific canonical threat vector."""
        feats = {f: 0.0 for f in FEATURE_ORDER}

        if scenario_type == "MID_CYCLE_BANK_DIVERSION":
            feats["vendor_bank_account_age_days"] = 1.5
            feats["is_new_bank_account_72h"] = 1.0
            feats["threshold_proximity_score"] = 0.95
            feats["invoice_amount_vs_90d_avg_ratio"] = 4.12
            feats["vendor_zscore_amount_last_90d"] = 3.5
            feats["rolling_24h_spend_vs_approval_limit"] = 0.98

        elif scenario_type == "SMURFING_SPLIT_INVOICE":
            feats["vendor_bank_account_age_days"] = 120.0
            feats["threshold_proximity_score"] = 0.98
            feats["smurfing_24h_pattern_detected"] = 1.0
            feats["rolling_24h_spend_vs_approval_limit"] = 1.85
            feats["invoice_amount_vs_90d_avg_ratio"] = 2.4

        elif scenario_type == "INTERNAL_COLLUSION":
            feats["degree_centrality_vendor"] = 0.45
            feats["shared_contact_attribute_flag"] = 1.0
            feats["po_amount_variance_ratio"] = 0.15
            feats["invoice_amount_vs_90d_avg_ratio"] = 3.0

        elif scenario_type == "SHELL_COMPANY":
            feats["vendor_bank_account_age_days"] = 5.0
            feats["shared_bank_account_cross_vendor_flag"] = 1.0
            feats["degree_centrality_vendor"] = 0.55
            feats["is_submitted_off_hours"] = 1.0

        elif scenario_type == "PHANTOM_DELIVERY":
            feats["po_amount_variance_ratio"] = 0.65
            feats["extraction_confidence"] = 0.70
            feats["invoice_amount_vs_90d_avg_ratio"] = 5.2

        return feats
