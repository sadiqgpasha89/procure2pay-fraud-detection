"""The single feature definition used for BOTH training and online scoring (no train/serve skew).

`FeatureBuilder.build(invoice)` evaluates everything *as of the invoice timestamp* using strictly
earlier history, so replaying an old invoice yields the same vector it would have had live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean, median, pstdev

from p2p.common.config import get_settings
from p2p.common.models import GoodsReceipt, Invoice, PurchaseOrder, Vendor
from p2p.data.repository import Warehouse
from p2p.features.graph import KnowledgeGraph

FEATURE_ORDER: list[str] = [
    # real-time / document
    "po_variance_ratio", "missing_po_flag", "goods_receipt_missing", "gr_coverage_ratio",
    "bank_age_days", "bank_changed_within_72h", "bank_changed_within_14d", "vendor_age_days",
    "threshold_proximity", "is_off_hours", "extraction_confidence", "round_amount_flag", "po_sod_violation",
    # behavioural
    "amount_zscore_vendor", "amount_ratio_vendor_mean", "vendor_history_count", "off_hours_ratio_30d",
    "rolling_24h_spend_ratio", "rolling_7d_spend_ratio", "prior_invoices_24h", "smurfing_pattern",
    "near_duplicate_flag", "exact_duplicate_count", "unit_price_ratio_vendor_hist", "approver_concentration",
    # graph
    "vendor_shared_bank_flag", "vendor_shared_address_flag", "vendor_employee_link_flag", "staff_vendor_link_flag",
    "vendor_degree",
]

DISPLAY = {  # human-readable names for SHAP drivers in dossiers
    "bank_changed_within_72h": "bank account changed within 72h", "bank_age_days": "days since bank change",
    "threshold_proximity": "proximity to $50k approval limit", "smurfing_pattern": "split-invoice pattern",
    "goods_receipt_missing": "no goods receipt", "staff_vendor_link_flag": "staff–vendor shared contact",
}


@dataclass
class TxContext:
    invoice: Invoice
    vendor: Vendor
    po: PurchaseOrder | None
    gr: GoodsReceipt | None
    features: dict[str, float]


def _is_off_hours(dt) -> bool:
    return dt.weekday() >= 5 or dt.hour < 7 or dt.hour >= 19


class FeatureBuilder:
    def __init__(self, wh: Warehouse, graph: KnowledgeGraph):
        self.wh, self.graph, self.s = wh, graph, get_settings()

    def load(self, invoice_id: str) -> TxContext | None:
        inv = self.wh.get_invoice(invoice_id)
        return self.build(inv) if inv else None

    def build(self, inv: Invoice) -> TxContext:
        wh, s = self.wh, self.s
        vendor = wh.get_vendor(inv.vendor_id)
        if vendor is None:
            raise LookupError(f"unknown vendor {inv.vendor_id}")
        po = wh.get_po(inv.po_id) if inv.po_id else None
        gr = wh.get_goods_receipt(inv.po_id) if inv.po_id else None
        if gr and gr.received_at > inv.invoice_date:
            gr = None  # not yet received when the invoice arrived
        t = inv.invoice_date
        f: dict[str, float] = {}

        # ── document / real-time ──
        f["missing_po_flag"] = float(po is None)
        f["po_variance_ratio"] = round((inv.total_amount - po.amount) / po.amount, 5) if po and po.amount > 0 else 0.0
        f["goods_receipt_missing"] = float(po is not None and gr is None)
        f["gr_coverage_ratio"] = round(min(2.0, gr.received_amount / inv.total_amount), 4) if gr and inv.total_amount > 0 else 0.0
        changes = wh.bank_changes(vendor.vendor_id, before=t)
        last_change = changes[-1].changed_at if changes else vendor.registration_date
        hours = max(0.0, (t - last_change).total_seconds() / 3600.0)
        f["bank_age_days"] = round(min(hours / 24.0, 365.0), 3)  # bounded: only *recent* change matters, and an unbounded age drifts with the calendar
        f["bank_changed_within_72h"] = float(bool(changes) and hours <= s.BANK_FREEZE_HOURS)
        f["bank_changed_within_14d"] = float(bool(changes) and hours <= 14 * 24)
        f["vendor_age_days"] = float(min(max(0, (t - vendor.registration_date).days), 730))
        lim = s.DUAL_APPROVAL_LIMIT
        f["threshold_proximity"] = round((inv.total_amount - 0.8 * lim) / (0.2 * lim), 4) if 0.8 * lim <= inv.total_amount < lim else 0.0
        f["is_off_hours"] = float(_is_off_hours(t))
        f["extraction_confidence"] = inv.extraction_confidence
        f["round_amount_flag"] = float(inv.total_amount >= 5000 and inv.total_amount % 1000 == 0)
        f["po_sod_violation"] = float(po is not None and po.approved_by is not None and po.created_by == po.approved_by)

        # ── behavioural (strictly prior history) ──
        hist365 = wh.vendor_invoices(vendor.vendor_id, t, since=t - timedelta(days=365), exclude=inv.invoice_id)  # one read, sliced below
        hist90 = [h for h in hist365 if h.invoice_date >= t - timedelta(days=90)]
        amounts = [h.total_amount for h in hist90]
        f["vendor_history_count"] = float(len(amounts))
        if len(amounts) >= 3:
            mu, sd = mean(amounts), max(pstdev(amounts), 0.05 * mean(amounts), 1.0)
            f["amount_zscore_vendor"] = round(max(-10.0, min(10.0, (inv.total_amount - mu) / sd)), 4)
            f["amount_ratio_vendor_mean"] = round(min(20.0, inv.total_amount / mu), 4)
        else:
            f["amount_zscore_vendor"], f["amount_ratio_vendor_mean"] = 0.0, 1.0
        hist30 = [h for h in hist90 if h.invoice_date >= t - timedelta(days=30)]
        f["off_hours_ratio_30d"] = round(sum(_is_off_hours(h.invoice_date) for h in hist30) / len(hist30), 4) if hist30 else 0.0
        w24 = [h for h in hist90 if h.invoice_date >= t - timedelta(hours=24) and h.cost_center == inv.cost_center]
        w7 = [h for h in hist90 if h.invoice_date >= t - timedelta(days=7) and h.cost_center == inv.cost_center]
        r24, r7 = sum(h.total_amount for h in w24) + inv.total_amount, sum(h.total_amount for h in w7) + inv.total_amount
        f["rolling_24h_spend_ratio"], f["rolling_7d_spend_ratio"] = round(r24 / lim, 4), round(r7 / lim, 4)
        f["prior_invoices_24h"] = float(len(w24))
        f["smurfing_pattern"] = float(inv.total_amount < lim and r24 >= lim and len(w24) > 0)
        same = wh.same_number_invoices(vendor.vendor_id, inv.invoice_number, inv.invoice_id, t)
        f["exact_duplicate_count"] = float(len([d for d in same if d.invoice_date <= t and (d.invoice_date, d.invoice_id) < (t, inv.invoice_id)]))
        near = [h for h in hist90 if h.invoice_date >= t - timedelta(days=7) and inv.total_amount > 0
                and abs(h.total_amount - inv.total_amount) / inv.total_amount < 0.01]
        f["near_duplicate_flag"] = float(bool(near) or f["exact_duplicate_count"] > 0)
        f["unit_price_ratio_vendor_hist"] = self._unit_price_ratio(inv, hist365)
        f["approver_concentration"] = self._approver_concentration(vendor.vendor_id, po, t)

        # ── graph ──
        f.update(self.graph.metrics(vendor.vendor_id, (po.created_by if po else None, po.approved_by if po else None, inv.submitted_by,
                                                       gr.received_by if gr else None)))
        return TxContext(inv, vendor, po, gr, {k: float(f[k]) for k in FEATURE_ORDER})

    @staticmethod
    def _unit_price_ratio(inv: Invoice, history: list[Invoice]) -> float:
        if not inv.lines:
            return 1.0
        item = inv.lines[0].item_code
        prior = [h.lines[0].unit_price for h in history if h.lines and h.lines[0].item_code == item and h.lines[0].unit_price > 0]
        if len(prior) < 2:
            return 1.0
        return round(min(5.0, inv.lines[0].unit_price / median(prior)), 4)

    def _approver_concentration(self, vendor_id: str, po: PurchaseOrder | None, t) -> float:
        if po is None or not po.approved_by:
            return 0.0
        pos = [p for p in self.wh.vendor_pos(vendor_id, t) if p.po_id != po.po_id]
        if len(pos) < 3:
            return 0.0
        return round(sum(p.approved_by == po.approved_by for p in pos) / len(pos), 4)


def is_finite(features: dict[str, float]) -> bool:
    return all(math.isfinite(v) for v in features.values())
