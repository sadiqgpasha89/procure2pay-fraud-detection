"""Real-time feature computation (<20ms latency domain)."""
from datetime import datetime
from typing import Dict, Optional
from src.core.models import Invoice, PurchaseOrder, Vendor


def compute_real_time_features(
    invoice: Invoice,
    vendor: Optional[Vendor] = None,
    po: Optional[PurchaseOrder] = None,
    reference_time: Optional[datetime] = None
) -> Dict[str, float]:
    """Compute instant streaming features from transaction payload."""
    ref_time = reference_time or datetime.utcnow()
    features: Dict[str, float] = {}

    # 1. Invoice vs PO Variance: abs(invoice_amount - po_amount) / po_amount
    if po and po.amount > 0:
        variance = abs(invoice.total_amount - po.amount) / po.amount
        features["po_amount_variance_ratio"] = round(variance, 4)
    else:
        # If PO is missing or zero, high variance flag
        features["po_amount_variance_ratio"] = 1.0 if not po else 0.0

    # 2. Bank Account Age & 72-Hour Modification Flag
    if vendor and vendor.bank_account_updated_at:
        delta = ref_time - vendor.bank_account_updated_at
        hours_since_bank_update = max(0.0, delta.total_seconds() / 3600.0)
        days_since_bank_update = hours_since_bank_update / 24.0
        features["vendor_bank_account_age_days"] = round(days_since_bank_update, 2)
        features["is_new_bank_account_72h"] = 1.0 if hours_since_bank_update <= 72.0 else 0.0
    else:
        features["vendor_bank_account_age_days"] = 365.0
        features["is_new_bank_account_72h"] = 0.0

    # 3. Threshold Proximity Score (Proximity to $50,000 dual-approver limit)
    # Attackers split or submit invoices at $48,000 - $49,999 to evade 2nd sign-off
    dual_limit = 50000.00
    if 40000.00 <= invoice.total_amount < dual_limit:
        proximity = 1.0 - ((dual_limit - invoice.total_amount) / 10000.00)
        features["threshold_proximity_score"] = round(max(0.0, proximity), 4)
    else:
        features["threshold_proximity_score"] = 0.0

    # 4. Weekend / Off-Hours Submission Flag
    inv_date = invoice.invoice_date or ref_time
    is_weekend = 1.0 if inv_date.weekday() >= 5 else 0.0
    is_off_hours = 1.0 if (inv_date.hour < 7 or inv_date.hour >= 19) else 0.0
    features["is_submitted_off_hours"] = 1.0 if (is_weekend or is_off_hours) else 0.0

    # 5. Extraction Confidence from OCR
    features["extraction_confidence"] = round(invoice.extraction_confidence, 4)

    return features
