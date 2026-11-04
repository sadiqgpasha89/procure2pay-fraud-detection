"""Deterministic rules for Tier 1 Rule-Based System (Strategy Pattern)."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.core.config import get_settings
from src.core.interfaces import RuleProtocol
from src.core.models import GoodsReceipt, Invoice, PurchaseOrder, Vendor


class ExactDuplicateInvoiceRule:
    """Detect duplicate invoice IDs submitted for the same vendor."""
    rule_name: str = "EXACT_DUPLICATE_INVOICE"
    description: str = "Duplicate invoice number submitted for identical vendor within lookback window."

    def __init__(self, historical_invoice_ids: Optional[List[str]] = None):
        self.seen_invoice_keys = set(historical_invoice_ids or [])

    def register_seen(self, vendor_id: str, invoice_id: str):
        self.seen_invoice_keys.add(f"{vendor_id.strip()}#{invoice_id.strip()}")

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        key = f"{invoice.vendor_id.strip()}#{invoice.invoice_id.strip()}"
        if key in self.seen_invoice_keys:
            return f"Duplicate invoice identifier detected: {invoice.invoice_id} for vendor {invoice.vendor_id}"
        return None


class BankChange72hFreezeRule:
    """Enforce mandatory payment freeze if vendor bank details modified <= 72h before payment (Edge Case 4)."""
    rule_name: str = "BANK_MODIFIED_WITHIN_72_HOURS"
    description: str = "Vendor settlement bank details were modified within 72 hours of invoice submission."

    def __init__(self, freeze_hours: Optional[int] = None):
        settings = get_settings()
        self.freeze_hours = freeze_hours or settings.BANK_MODIFICATION_FREEZE_HOURS

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not vendor or not vendor.bank_account_updated_at:
            return None

        ref_time = invoice.invoice_date or datetime.utcnow()
        delta = ref_time - vendor.bank_account_updated_at
        hours = delta.total_seconds() / 3600.0

        if 0.0 <= hours <= self.freeze_hours:
            return (
                f"Mandatory freeze: Settlement bank account modified {hours:.1f} hours ago "
                f"(<= {self.freeze_hours}h limit). Requires dual-authorization before release."
            )
        return None


class SanctionsOFACRule:
    """Screen vendor against OFAC / PEP sanctions lists."""
    rule_name: str = "SANCTIONS_OFAC_HIT"
    description: str = "Vendor or entity executive appears on active sanction or PEP watchlist."

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if vendor and vendor.is_sanctioned:
            return f"Statutory block: Vendor {vendor.vendor_id} ({vendor.legal_name}) is on active sanction list."
        return None


class POCeilingToleranceRule:
    """Enforce hard variance ceiling between invoice total and PO amount."""
    rule_name: str = "PO_CEILING_TOLERANCE_EXCEEDED"
    description: str = "Invoice amount exceeds linked PO committed amount beyond allowable tolerance."

    def __init__(self, tolerance_pct: Optional[float] = None):
        settings = get_settings()
        self.tolerance_pct = tolerance_pct or settings.PO_VARIANCE_TOLERANCE_PCT

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not po or po.amount <= 0:
            return None

        variance = (invoice.total_amount - po.amount) / po.amount
        if variance > self.tolerance_pct:
            return (
                f"Hard ceiling exceeded: Invoice amount (${invoice.total_amount:,.2f}) exceeds PO amount "
                f"(${po.amount:,.2f}) by {variance * 100:.1f}% (allowable ceiling: {self.tolerance_pct * 100:.1f}%)."
            )
        return None


class SplitInvoicingHardRule:
    """Hard check on sequential split invoices under $50k approval limit."""
    rule_name: str = "THRESHOLD_PROXIMITY_SPLIT_DETECTED"
    description: str = "Invoice submitted just below approval threshold with matching companion invoices."

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        ctx = context or {}
        if ctx.get("smurfing_24h_pattern_detected", 0.0) == 1.0:
            return (
                f"Smurfing pattern detected: Current invoice (${invoice.total_amount:,.2f}) is under $50,000 threshold "
                f"but 24-hour vendor cumulative spend exceeds dual-approval limit."
            )
        return None
