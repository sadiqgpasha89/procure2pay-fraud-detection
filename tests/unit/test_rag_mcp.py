import asyncio

import pytest

from p2p.rag.index import HybridIndex, chunk_document
from p2p.common.models import Document


@pytest.fixture(scope="module")
def rag(world, wh):
    from p2p.rag.index import get_index
    return get_index(force_reload=True)


def test_chunking_keeps_short_docs_whole_and_overlaps_long_ones():
    short = Document(doc_id="D1", doc_type="POLICY", title="t", text="Short policy.")
    assert [c.chunk_id for c in chunk_document(short)] == ["D1#0"]
    long = Document(doc_id="D2", doc_type="CONTRACT", title="t", text=". ".join(f"Sentence number {i} about pricing" for i in range(80)))
    cs = chunk_document(long, max_chars=300)
    assert len(cs) > 3 and all(len(c.text) <= 400 for c in cs)


def test_retrieval_is_vendor_scoped(rag, wh):
    v1, v2 = [v for v in wh.vendors() if v.vendor_id.startswith("VEND")][:2]
    hits = rag.search("unit price cap pricing", k=8, vendor_id=v1.vendor_id, doc_types=["CONTRACT"])
    assert hits and all(h.chunk.vendor_id == v1.vendor_id for h in hits)
    assert v2.vendor_id not in {h.chunk.vendor_id for h in hits}


def test_retrieval_finds_relevant_policy_and_lookalike_email(rag, wh, labels, by_type):
    assert rag.search("splitting invoices below approval limit", 2, doc_types=["POLICY"])[0].chunk.doc_id == "POL-002"
    inv = wh.get_invoice(by_type["BANK_DIVERSION"][0])
    emails = rag.search("urgent bank account change remittance", 3, vendor_id=inv.vendor_id, doc_types=["EMAIL"])
    assert emails and any("URGENT" in h.chunk.text for h in emails)


def _tool(fn):
    return getattr(fn, "fn", fn)


def test_erp_tools_are_point_in_time_and_evidence_verifies(world, by_type, monkeypatch):
    from p2p.mcp_servers import erp_server as erp
    monkeypatch.setattr(erp, "_wh", None)
    iid = by_type["PHANTOM_DELIVERY"][0]
    ctx = _tool(erp.get_transaction_context)(iid)
    assert ctx["invoice"]["bank_iban"].count("*") >= 4 and "tax" not in str(ctx["vendor"]["tax_id"]).lower()  # PII masked
    for e in ctx["evidence"]:
        assert _tool(erp.verify_evidence)(e["record_id"], e["verification_hash"])["valid"] is True
        assert _tool(erp.verify_evidence)(e["record_id"], "0" * 64)["valid"] is False
    assert _tool(erp.verify_evidence)("INV:DOES-NOT-EXIST", "0" * 64)["valid"] is False
    rel = _tool(erp.find_related_invoices)(by_type["DUPLICATE"][0])
    assert "error" not in rel and (rel["same_invoice_number"] or rel["same_amount_within_7d"])


def test_agents_can_never_reach_ground_truth_labels():
    """Structural guarantee: the label store is imported only by training/evaluation code — never by tools or agents."""
    import inspect
    import p2p.agents.investigator as inv
    import p2p.agents.reference_llm as ref
    import p2p.agents.specialists as spec
    from p2p.mcp_servers import erp_server, graph_server, knowledge_server
    for mod in (erp_server, graph_server, knowledge_server, inv, ref, spec):
        src = inspect.getsource(mod)
        assert "LabelStore" not in src and "FROM labels" not in src, mod.__name__


def test_graph_tool_links_staff_and_redacts(world, by_type, monkeypatch):
    from p2p.mcp_servers import graph_server as g
    monkeypatch.setattr(g, "_state", {})
    found = None
    for iid in by_type["COLLUSION"]:
        r = _tool(g.get_entity_links)(iid)
        if any(l["roles_on_this_transaction"] for l in r["links"] if l["linked_entity_kind"] == "EMPLOYEE"):
            found = r; break
    assert found and found["entity_conflicts_detected"]
    assert all(l["shared_value"] == "<redacted>" for l in found["links"] if l["linked_entity_kind"] == "EMPLOYEE")
    e = found["evidence"][0]
    assert _tool(g.verify_evidence)(e["record_id"], e["verification_hash"])["valid"]


def test_knowledge_tool_neutralises_planted_injection(world, wh, by_type, monkeypatch):
    from p2p.mcp_servers import knowledge_server as k
    docs = [d for d in wh.documents("EMAIL") if "SYSTEM NOTICE" in d.text]
    assert docs, "synthetic corpus must contain at least one planted injection"
    r = _tool(k.search_documents)("bank account change request urgent", vendor_id=docs[0].vendor_id, doc_types=["EMAIL"])
    flagged = [h for h in r["hits"] if h["injection_suspected"]]
    assert flagged and all("ignore all previous" not in h["text"].lower() and "AUTO_APPROVE" not in h["text"] for h in flagged)
    e = r["evidence"][0]
    assert _tool(k.verify_evidence)(e["record_id"], e["verification_hash"])["valid"]
