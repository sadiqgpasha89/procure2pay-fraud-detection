"""Behavioral and rolling-window feature computation (Batch / Redis aggregated domain)."""
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from src.core.models import Invoice


class HistoricalSpendProfile:
    """Historical transaction spend profile for a vendor."""

    def __init__(
        self,
        vendor_id: str,
        amounts_90d: Optional[List[float]] = None,
        off_hours_submissions_30d: int = 0,
        total_submissions_30d: int = 1,
    ):
        self.vendor_id = vendor_id
        self.amounts_90d = amounts_90d
        self.off_hours_submissions_30d = off_hours_submissions_30d
        self.total_submissions_30d = max(1, total_submissions_30d)

    def get_mean(self, fallback_amount: float = 5000.0) -> float:
        if self.amounts_90d:
            return sum(self.amounts_90d) / max(1, len(self.amounts_90d))
        return fallback_amount

    def get_std(self, fallback_amount: float = 5000.0) -> float:
        if self.amounts_90d and len(self.amounts_90d) > 1:
            m = self.get_mean(fallback_amount)
            variance = sum((x - m) ** 2 for x in self.amounts_90d) / len(self.amounts_90d)
            return math.sqrt(variance) if variance > 0 else 1.0
        return max(1.0, fallback_amount * 0.1)


def compute_behavioral_features(
    invoice: Invoice,
    history: Optional[HistoricalSpendProfile] = None,
    rolling_24h_spend: float = 0.0,
    rolling_7d_spend: float = 0.0,
    rolling_30d_spend: float = 0.0,
) -> Dict[str, float]:
    """Calculate statistical behavioral features and smurfing indicators."""
    profile = history or HistoricalSpendProfile(vendor_id=invoice.vendor_id)
    features: Dict[str, float] = {}

    # 1. Z-Score of current invoice amount relative to 90-day history
    mean = profile.get_mean(invoice.total_amount)
    std = profile.get_std(invoice.total_amount)
    zscore = (invoice.total_amount - mean) / (std if std > 0 else 1.0)
    features["vendor_zscore_amount_last_90d"] = round(zscore, 4)

    # 2. Ratio of current invoice amount to 90-day average
    ratio = invoice.total_amount / (mean if mean > 0 else 1.0)
    features["invoice_amount_vs_90d_avg_ratio"] = round(ratio, 4)

    # 3. Off-Hours Submission Ratio (30 Days)
    off_hours_ratio = profile.off_hours_submissions_30d / profile.total_submissions_30d
    features["off_hours_submission_ratio_30d"] = round(off_hours_ratio, 4)

    # 4. Smurfing / Split-Invoicing Rolling Spend Features (Edge Case 6)
    dual_approval_limit = 50000.00
    cumulative_24h = rolling_24h_spend + invoice.total_amount
    cumulative_7d = rolling_7d_spend + invoice.total_amount
    cumulative_30d = rolling_30d_spend + invoice.total_amount

    features["rolling_24h_spend_vs_approval_limit"] = round(cumulative_24h / dual_approval_limit, 4)
    features["rolling_7d_spend_vs_approval_limit"] = round(cumulative_7d / dual_approval_limit, 4)
    features["rolling_30d_spend_vs_approval_limit"] = round(cumulative_30d / dual_approval_limit, 4)

    # Smurfing flag: individual invoice under limit, but cumulative 24h aggregate with companion invoices exceeds limit!
    is_smurfing = (
        1.0 if (invoice.total_amount < dual_approval_limit and cumulative_24h >= dual_approval_limit and rolling_24h_spend > 0)
        else 0.0
    )
    features["smurfing_24h_pattern_detected"] = is_smurfing

    return features
