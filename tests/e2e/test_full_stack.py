"""Boots all 10 microservices as real processes (with SERVICE_TOKEN set, so every REST/MCP/A2A hop must authenticate)
and drives the gateway like an ERP payment run would."""
import socket

import httpx
import pytest

from p2p.common.config import reset_settings_cache

pytestmark = pytest.mark.e2e
TOKEN = "e2e-shared-secret"


def _port_busy(p: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) == 0


@pytest.fixture(scope="module")
def stack(world):
    if any(_port_busy(p) for p in (8100, 8101, 8102, 8103, 8201, 8202, 8203, 8300, 8301, 8302)):
        pytest.skip("a local stack is already running on the service ports (run `python -m p2p.stack down`)")
    import os
    os.environ["SERVICE_TOKEN"] = TOKEN
    reset_settings_cache()
    from p2p import stack as st
    assert st.up(timeout=120) == 0, "stack failed to become healthy"
    yield
    st.down()
    os.environ.pop("SERVICE_TOKEN", None)
    reset_settings_cache()


@pytest.fixture(scope="module")
def h(stack):
    with httpx.Client(timeout=120, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        yield c


def ev(h, iid, wait=True):
    r = h.post("http://127.0.0.1:8100/v1/evaluate", json={"invoice_id": iid, "wait_for_investigation": wait})
    assert r.status_code == 200, r.text
    return r.json()


def test_service_auth_is_enforced(stack):
    assert httpx.post("http://127.0.0.1:8100/v1/evaluate", json={"invoice_id": "INV-000001"}).status_code == 401
    assert httpx.get("http://127.0.0.1:8203/mcp").status_code == 401
    assert httpx.get("http://127.0.0.1:8100/health").status_code == 200
    assert httpx.get("http://127.0.0.1:8300/.well-known/agent-card.json").json()["name"] == "investigator"  # discovery stays public


def test_clean_invoice_is_auto_approved_without_a_case(h, engine, by_type):
    iid = next(i for i in by_type["LEGIT"][-80:] if engine.score(i).route.value == "APPROVE")
    d = ev(h, iid, wait=False)
    assert d["final_action"] == "AUTO_APPROVE" and d["case_id"] is None and not d["requires_human_signoff"] and d["total_latency_ms"] < 850
    assert ev(h, iid, wait=False)["idempotency_key"] == d["idempotency_key"]  # idempotent replay


def test_sanctioned_vendor_is_blocked_and_frozen(h, by_type):
    d = ev(h, by_type["SANCTIONED_VENDOR"][0])
    assert d["route"] == "BLOCK" and d["final_action"] == "BLOCK_TRANSACTION"
    assert d["dossier"]["recommended_action"] == "FREEZE_VENDOR" and d["dossier"]["requires_human_signoff"]


def test_collusion_is_investigated_over_a2a_with_verified_graph_evidence(h, engine, by_type):
    iid = next(i for i in by_type["COLLUSION"] if engine.score(i).tier2.composite_score >= 0.35)
    d = ev(h, iid)
    dos = d["dossier"]
    assert d["final_action"] in ("HOLD_PAYMENT", "BLOCK_TRANSACTION") and dos
    assert dos["recommended_action"] == "HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT"
    assert dos["graph_findings"]["entity_conflicts_detected"] and any(e["source"] == "GRAPH_DB" for e in dos["evidence_manifest"])
    assert "[Evidence:" in dos["agent_narrative"] and dos["synthesis_latency_sec"] < 8


def test_prompt_injection_in_vendor_email_cannot_steer_the_outcome(h, wh, by_type):
    bad = {d.vendor_id for d in wh.documents("EMAIL") if "SYSTEM NOTICE" in d.text}
    iid = next(i for i in by_type["BANK_DIVERSION"] if wh.get_invoice(i).vendor_id in bad)
    dos = ev(h, iid)["dossier"]
    assert dos["recommended_action"] != "AUTO_APPROVE"
    assert any("instruction-like" in f for f in dos["document_findings"])
    assert "ignore all previous" not in dos["agent_narrative"].lower()


def test_audit_ledger_is_intact_and_records_decisions_dossiers_and_human_disposition(h):
    cases = h.get("http://127.0.0.1:8102/v1/cases").json()
    assert cases
    done = next(c for c in cases if c["status"] == "PENDING_REVIEW")
    r = h.post(f"http://127.0.0.1:8102/v1/cases/{done['case_id']}/disposition",
               json={"decision": "CONFIRMED_FRAUD", "analyst_id": "e2e", "rationale": "test", "accepted_recommendation": True})
    assert r.status_code == 200
    v = h.get("http://127.0.0.1:8103/v1/verify").json()
    assert v["valid"] and v["blocks"] >= 8
    kinds = {b["event_type"] for b in h.get("http://127.0.0.1:8103/v1/entries?limit=500").json()}
    assert {"CASE_DOSSIER_COMPILED", "HUMAN_DISPOSITION"} <= kinds and any(k.startswith("DECISION_") for k in kinds)
