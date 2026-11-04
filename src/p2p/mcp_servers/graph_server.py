"""mcp-graph — read-only entity-link analysis over the vendor/employee knowledge graph.

Answers "does this vendor share a bank account, address or phone number with another vendor or with the staff who
created, approved, keyed or received this transaction?" Edges exist only on exact normalised attribute matches.
"""
from __future__ import annotations

from fastmcp import FastMCP

from p2p.common.evidence import content_hash, make_evidence
from p2p.data.repository import Warehouse, open_warehouse
from p2p.features.graph import KnowledgeGraph
from p2p.mcp_servers.common import build_asgi

mcp = FastMCP("p2p-graph")
_state: dict[str, object] = {}


def wh() -> Warehouse:
    if "wh" not in _state:
        _state["wh"] = open_warehouse()
    return _state["wh"]  # type: ignore[return-value]


def kg() -> KnowledgeGraph:
    if "kg" not in _state:
        _state["kg"] = KnowledgeGraph.from_warehouse(wh())
    return _state["kg"]  # type: ignore[return-value]


def _link_payload(link) -> dict:
    return {"type": link.attribute_type, "value": link.attribute, "a": link.entity_a, "b": link.entity_b}


@mcp.tool
def get_entity_links(invoice_id: str) -> dict:
    """Analyse the knowledge graph around the vendor of this invoice. Returns every shared attribute (bank account, address,
    phone, director) between the vendor and other vendors or employees, and flags links to the staff who touched THIS
    transaction (PO creator, PO approver, AP clerk, goods receiver). Use to detect shell companies and collusion."""
    w = wh()
    inv = w.get_invoice(invoice_id)
    if inv is None:
        return {"error": f"invoice {invoice_id} not found"}
    po = w.get_po(inv.po_id) if inv.po_id else None
    gr = w.get_goods_receipt(inv.po_id) if inv.po_id else None
    roles = {}
    for role, emp in (("po_creator", po.created_by if po else None), ("po_approver", po.approved_by if po else None),
                      ("ap_clerk", inv.submitted_by), ("goods_receiver", gr.received_by if gr else None)):
        if emp:
            roles.setdefault(emp, []).append(role)
    staff = tuple(roles)
    la = kg().link_analysis(inv.vendor_id)
    findings = kg().findings(inv.vendor_id, staff)
    links = []
    for link in la.links:
        e = w.get_employee(link.entity_b) if link.entity_b.startswith("EMP") else None
        links.append({"record_id": link.record_id, "attribute_type": link.attribute_type, "shared_value": "<redacted>" if (link.attribute_type in ("BANK", "PHONE") or e) else link.attribute,
                      "linked_entity": link.entity_b, "linked_entity_kind": "EMPLOYEE" if e else "VENDOR",
                      "linked_employee_role": e.role if e else None, "linked_employee_name": e.name if e else None,
                      "roles_on_this_transaction": roles.get(link.entity_b, [])})
    return {"vendor_id": inv.vendor_id, "transaction_staff": {k: v for k, v in roles.items()}, "entity_conflicts_detected": findings.entity_conflicts_detected,
            "links": links, "summary": findings.details, "collusion_nodes": findings.collusion_nodes,
            "evidence": [make_evidence("GRAPH_DB", link.record_id, _link_payload(link)).model_dump() for link in la.links]}


@mcp.tool
def verify_evidence(record_id: str, verification_hash: str) -> dict:
    """Re-derive the content hash of a graph edge (record id `EDGE-<TYPE>-<vendor>-<entity>`) and compare with the claimed hash."""
    parts = record_id.split("-", 2)
    if len(parts) < 3 or parts[0] != "EDGE":
        return {"valid": False, "reason": "malformed record id"}
    _, _, rest = parts
    vendor = "-".join(rest.split("-")[:2])
    for link in kg().link_analysis(vendor).links:
        if link.record_id == record_id:
            ok = content_hash(_link_payload(link)) == verification_hash
            return {"valid": ok, "reason": "" if ok else "hash mismatch"}
    return {"valid": False, "reason": "edge not found"}


app = build_asgi(mcp, "mcp-graph")
