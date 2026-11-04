"""mcp-erp — READ-ONLY ERP / warehouse tools for investigator agents (Model Context Protocol).

There is deliberately no write tool and no free-form query tool: agents can only call these typed,
parameterised functions (spec §5.4 guardrail 1). Every tool is point-in-time relative to the invoice
under investigation, and every record it returns is accompanied by content-hashed `evidence`.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastmcp import FastMCP

from p2p.common.evidence import content_hash, make_evidence
from p2p.common.security import mask_iban, mask_phone, mask_tax_id
from p2p.data.repository import Warehouse, open_warehouse
from p2p.mcp_servers.common import build_asgi

mcp = FastMCP("p2p-erp")
_wh: Warehouse | None = None


def wh() -> Warehouse:
    global _wh
    _wh = _wh or open_warehouse()
    return _wh


def _ev(source_id: str, record: Any) -> dict:
    return make_evidence("ERP", source_id, record).model_dump()


def _masked_vendor(v) -> dict:
    d = v.model_dump(mode="json")
    d.update(tax_id=mask_tax_id(v.tax_id), bank_iban=mask_iban(v.bank_iban), phone=mask_phone(v.phone))
    return d


def _masked_invoice(i) -> dict:
    d = i.model_dump(mode="json")
    d.update(bank_iban=mask_iban(i.bank_iban), vendor_tax_id=mask_tax_id(i.vendor_tax_id))
    return d


@mcp.tool
def get_transaction_context(invoice_id: str) -> dict:
    """Fetch the invoice with its purchase order, goods receipt and vendor master record, plus a computed
    three-way match (PO variance, goods-receipt coverage, segregation-of-duties). Use first when auditing an invoice."""
    w = wh()
    inv = w.get_invoice(invoice_id)
    if inv is None:
        return {"error": f"invoice {invoice_id} not found"}
    po = w.get_po(inv.po_id) if inv.po_id else None
    gr = w.get_goods_receipt(inv.po_id) if inv.po_id else None
    vendor = w.get_vendor(inv.vendor_id)
    evidence = [_ev(f"INV:{inv.invoice_id}", inv)]
    match: dict[str, Any] = {"po_present": po is not None, "goods_receipt_present": gr is not None and gr.received_at <= inv.invoice_date}
    if po:
        evidence.append(_ev(f"PO:{po.po_id}", po))
        match.update(po_amount=po.amount, invoice_total=inv.total_amount, po_variance_pct=round(100 * (inv.total_amount - po.amount) / po.amount, 2),
                     segregation_of_duties_violation=po.created_by == po.approved_by, po_created_by=po.created_by, po_approved_by=po.approved_by)
    if gr:
        evidence.append(_ev(f"GR:{gr.gr_id}", gr))
        match.update(goods_received_amount=gr.received_amount, goods_receipt_coverage_pct=round(100 * gr.received_amount / inv.total_amount, 1))
    if vendor:
        evidence.append(_ev(f"VENDOR:{vendor.vendor_id}", vendor))
    return {"invoice": _masked_invoice(inv), "purchase_order": po.model_dump(mode="json") if po else None,
            "goods_receipt": gr.model_dump(mode="json") if gr else None, "vendor": _masked_vendor(vendor) if vendor else None,
            "three_way_match": match, "evidence": evidence}


@mcp.tool
def get_vendor_invoice_history(invoice_id: str, months: int = 12, limit: int = 10) -> dict:
    """Summarise this vendor's invoicing in the `months` before the given invoice (count, mean, max, off-hours share) and
    list the most recent `limit` prior invoices. Use to judge whether the amount/timing is out of pattern."""
    w = wh()
    inv = w.get_invoice(invoice_id)
    if inv is None:
        return {"error": f"invoice {invoice_id} not found"}
    hist = w.vendor_invoices(inv.vendor_id, inv.invoice_date, since=inv.invoice_date - timedelta(days=30 * months), exclude=invoice_id)
    amounts = [h.total_amount for h in hist]
    recent = hist[-limit:]
    summary = {"vendor_id": inv.vendor_id, "prior_invoice_count": len(hist),
               "mean_amount": round(sum(amounts) / len(amounts), 2) if amounts else None, "max_amount": max(amounts) if amounts else None,
               "current_vs_mean_ratio": round(inv.total_amount / (sum(amounts) / len(amounts)), 2) if amounts else None,
               "off_hours_share": round(sum(h.invoice_date.weekday() >= 5 or h.invoice_date.hour < 7 or h.invoice_date.hour >= 19 for h in hist) / len(hist), 3) if hist else None}
    return {"summary": summary, "recent_invoices": [{"invoice_id": h.invoice_id, "invoice_number": h.invoice_number, "date": h.invoice_date.isoformat(),
                                                    "total_amount": h.total_amount, "cost_center": h.cost_center} for h in recent],
            "evidence": [_ev(f"INV:{h.invoice_id}", h) for h in recent]}


@mcp.tool
def get_bank_change_log(invoice_id: str) -> dict:
    """List this vendor's bank-detail changes up to the invoice date, with how many hours before the invoice each happened and
    how it was requested (portal / e-mail / phone) and from which sender domain."""
    w = wh()
    inv = w.get_invoice(invoice_id)
    if inv is None:
        return {"error": f"invoice {invoice_id} not found"}
    changes = w.bank_changes(inv.vendor_id, before=inv.invoice_date)
    return {"vendor_id": inv.vendor_id, "invoice_date": inv.invoice_date.isoformat(),
            "changes": [{"change_id": c.change_id, "changed_at": c.changed_at.isoformat(), "hours_before_invoice": round((inv.invoice_date - c.changed_at).total_seconds() / 3600, 1),
                         "old_iban": mask_iban(c.old_iban), "new_iban": mask_iban(c.new_iban), "requested_via": c.requested_via, "requested_from": c.requested_from}
                        for c in changes],
            "registered_vendor_domain": (w.get_vendor(inv.vendor_id).email_domain if w.get_vendor(inv.vendor_id) else None),
            "evidence": [_ev(f"BANKCHG:{c.change_id}", c) for c in changes]}


@mcp.tool
def find_related_invoices(invoice_id: str) -> dict:
    """Find other invoices from the same vendor that could be duplicates (same number, or same amount within 7 days) or split
    companions (same cost centre within 24 hours)."""
    w = wh()
    inv = w.get_invoice(invoice_id)
    if inv is None:
        return {"error": f"invoice {invoice_id} not found"}
    t = inv.invoice_date
    prior = w.vendor_invoices(inv.vendor_id, t, since=t - timedelta(days=7), exclude=invoice_id)
    dup_num = [i for i in w.same_number_invoices(inv.vendor_id, inv.invoice_number, invoice_id, t)]
    near = [i for i in prior if abs(i.total_amount - inv.total_amount) / max(inv.total_amount, 1) < 0.01 and i not in dup_num]
    comp = [i for i in prior if i.invoice_date >= t - timedelta(hours=24) and i.cost_center == inv.cost_center]
    total_24h = round(inv.total_amount + sum(i.total_amount for i in comp), 2)
    row = lambda i: {"invoice_id": i.invoice_id, "invoice_number": i.invoice_number, "date": i.invoice_date.isoformat(), "total_amount": i.total_amount}  # noqa: E731
    return {"same_invoice_number": [row(i) for i in dup_num], "same_amount_within_7d": [row(i) for i in near],
            "companions_within_24h_same_cost_center": [row(i) for i in comp], "combined_24h_total_including_current": total_24h,
            "dual_approval_limit": 50_000.0,
            "evidence": [_ev(f"INV:{i.invoice_id}", i) for i in {x.invoice_id: x for x in (*dup_num, *near, *comp)}.values()]}


def _resolve(record_id: str) -> Any | None:
    w = wh()
    kind, _, key = record_id.partition(":")
    return {"INV": w.get_invoice, "PO": w.get_po, "GR": lambda k: _gr(w, k), "VENDOR": w.get_vendor,
            "BANKCHG": lambda k: next((b for v in w.vendors() for b in w.bank_changes(v.vendor_id) if b.change_id == k), None)}.get(kind, lambda _: None)(key)


def _gr(w: Warehouse, gr_id: str):
    with w._conn() as c:  # noqa: SLF001 - read-only lookup by primary key
        row = c.execute("SELECT body FROM goods_receipts WHERE gr_id=?", (gr_id,)).fetchone()
    from p2p.common.models import GoodsReceipt
    return GoodsReceipt.model_validate_json(row["body"]) if row else None


@mcp.tool
def verify_evidence(record_id: str, verification_hash: str) -> dict:
    """Re-derive the content hash of an ERP record and compare it with the claimed hash. Returns {valid: bool}."""
    rec = _resolve(record_id)
    if rec is None:
        return {"valid": False, "reason": "record not found"}
    ok = content_hash(rec) == verification_hash
    return {"valid": ok, "reason": "" if ok else "hash mismatch"}


app = build_asgi(mcp, "mcp-erp")
