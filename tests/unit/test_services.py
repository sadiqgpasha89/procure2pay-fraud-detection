import pytest
from fastapi.testclient import TestClient

from p2p.common.config import reset_settings_cache
from p2p.common.models import Route


@pytest.fixture()
def scoring_client(world):
    from p2p.services.scoring.app import app
    with TestClient(app) as c:
        yield c


def test_scoring_service_routes_each_scenario(scoring_client, by_type, labels):
    def route(iid):
        r = scoring_client.post("/v1/score", json={"invoice_id": iid}); assert r.status_code == 200; return r.json()
    assert route(by_type["SANCTIONED_VENDOR"][0])["route"] == Route.BLOCK.value
    assert route(by_type["SHELL_VENDOR"][0])["route"] in (Route.INVESTIGATE.value, Route.BLOCK.value)
    legit_routes = [route(i)["route"] for i in by_type["LEGIT"][-60:]]
    assert legit_routes.count("APPROVE") / len(legit_routes) > 0.85
    assert scoring_client.post("/v1/score", json={"invoice_id": "INV-NOPE"}).status_code == 404
    assert scoring_client.get("/health").json()["status"] == "ok" and scoring_client.get("/v1/model").json()["data"]["synthetic"]


def test_scoring_p99_latency_budget(scoring_client, by_type):
    import numpy as np, time
    ts = []
    for iid in by_type["LEGIT"][-120:]:
        t = time.perf_counter(); scoring_client.post("/v1/score", json={"invoice_id": iid}); ts.append((time.perf_counter() - t) * 1000)
    assert np.percentile(ts, 99) < 850.0  # spec §7.4 (sequential, in-process; the e2e benchmark measures it under load)


def test_scoring_requires_bearer_token_when_configured(world, monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "s3cret"); reset_settings_cache()
    try:
        from p2p.common.ops import BearerAuthMiddleware
        from p2p.services.scoring.app import app
        app.middleware_stack = None  # rebuild with the token in effect
        app.user_middleware = [m for m in app.user_middleware if m.cls is not BearerAuthMiddleware]
        app.add_middleware(BearerAuthMiddleware, token="s3cret")
        with TestClient(app) as c:
            assert c.post("/v1/score", json={"invoice_id": "INV-000001"}).status_code == 401
            assert c.post("/v1/score", json={"invoice_id": "INV-000001"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert c.get("/health").status_code == 200  # probes stay open
            assert c.post("/v1/score", json={"invoice_id": "INV-000001"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    finally:
        monkeypatch.delenv("SERVICE_TOKEN"); reset_settings_cache()


def test_case_service_lifecycle_and_kpis(world, engine, by_type, monkeypatch, tmp_path):
    monkeypatch.setenv("CASE_DB_URL", f"sqlite:///{tmp_path}/cases.sqlite")  # isolated from any e2e run sharing the session dir
    monkeypatch.setenv("AUDIT_URL", "http://127.0.0.1:1")  # audit unreachable: disposition must still succeed
    reset_settings_cache()
    from p2p.common.models import CaseDossier, Action, RiskAssessment, RiskClass, TransactionIdentifiers, GraphFindings
    from p2p.services.case.app import app
    sc = engine.score(by_type["SHELL_VENDOR"][0])
    with TestClient(app) as c:
        body = {"invoice_id": sc.invoice_id, "vendor_id": sc.vendor_id, "amount": sc.amount, "scoring": sc.model_dump(mode="json")}
        case = c.post("/v1/cases", json=body).json()
        assert c.post("/v1/cases", json=body).json()["case_id"] == case["case_id"]  # idempotent per invoice
        assert c.post(f"/v1/cases/{case['case_id']}/disposition", json={"decision": "CONFIRMED_FRAUD", "analyst_id": "a", "rationale": "r"}).status_code == 409
        d = CaseDossier(case_id=case["case_id"], timestamp="2026-01-01T00:00:00Z", transaction_identifiers=TransactionIdentifiers(invoice_id=sc.invoice_id, vendor_id=sc.vendor_id),
                        risk_assessment=RiskAssessment(composite_risk_score=0.9, classification=RiskClass.HIGH), graph_findings=GraphFindings(), agent_narrative="n",
                        recommended_action=Action.HOLD_PAYMENT)
        assert c.put(f"/v1/cases/{case['case_id']}/dossier", json=d.model_dump(mode="json", by_alias=True)).json()["status"] == "PENDING_REVIEW"
        assert c.post(f"/v1/cases/{case['case_id']}/disposition", json={"decision": "BOGUS", "analyst_id": "a", "rationale": "r"}).status_code == 422
        out = c.post(f"/v1/cases/{case['case_id']}/disposition", json={"decision": "CONFIRMED_FRAUD", "analyst_id": "a", "rationale": "r", "accepted_recommendation": True})
        assert out.status_code == 200 and out.json()["status"] == "DISPOSED"
        assert c.post(f"/v1/cases/{case['case_id']}/disposition", json={"decision": "FALSE_POSITIVE", "analyst_id": "a", "rationale": "r"}).status_code == 409
        m = c.get("/v1/metrics").json()
        assert m["gross_prevented_loss"] == pytest.approx(sc.amount) and m["recommendation_acceptance_rate"] == 1.0
        assert "Investigator console" in c.get("/").text
    reset_settings_cache()
