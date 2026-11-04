"""API integration tests verifying all REST endpoints with httpx AsyncClient."""
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_and_readiness_probes():
    """Verify Kubernetes health probes return 200 OK."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_health = await client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "HEALTHY"

        res_ready = await client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "READY"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluate_endpoint_clean_transaction(clean_invoice, clean_vendor, sample_po):
    """POST /api/v1/evaluate should return Auto-Approval for clean invoice."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "invoice": clean_invoice.model_dump(mode="json"),
            "vendor": clean_vendor.model_dump(mode="json"),
            "purchase_order": sample_po.model_dump(mode="json"),
        }
        resp = await client.post("/api/v1/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_action"] == "AUTO_APPROVE"
        assert not data["requires_human_signoff"]
        assert data["tier1_result"]["is_blocked"] is False
        assert data["audit_hash"] != ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ocr_ingestion_and_routing_endpoint():
    """POST /api/v1/ingest/ocr should parse fields and report confidence."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "raw_text": "Invoice: INV-4401 Vendor ID: VEND-99 Total Amount: $1,250.00",
            "base_confidence": 0.95,
        }
        resp = await client.post("/api/v1/ingest/ocr", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["invoice"]["invoice_id"] == "INV-4401"
        assert data["overall_confidence"] > 0.50


@pytest.mark.integration
@pytest.mark.asyncio
async def test_case_management_and_disposition_workflow(clean_invoice, clean_vendor, sample_po):
    """Test full HITL flow: suspicious invoice generates case -> retrieved via API -> disposed."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Submit high risk invoice to generate case
        suspicious_inv = clean_invoice.model_copy(update={
            "invoice_id": "INV-API-SUSPECT-1",
            "total_amount": 49900.00,
        })
        matching_po = sample_po.model_copy(update={"amount": 50000.00})
        payload = {
            "invoice": suspicious_inv.model_dump(mode="json"),
            "vendor": clean_vendor.model_dump(mode="json"),
            "purchase_order": matching_po.model_dump(mode="json"),
        }
        eval_resp = await client.post("/api/v1/evaluate", json=payload)
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["case_dossier"] is not None
        case_id = eval_data["case_dossier"]["case_id"]

        # 2. Query case via GET /api/v1/cases/{case_id}
        case_resp = await client.get(f"/api/v1/cases/{case_id}")
        assert case_resp.status_code == 200
        case_data = case_resp.json()
        assert case_data["status"] == "PENDING_REVIEW"
        assert case_data["dossier"]["case_id"] == case_id

        # 3. Post Human Disposition
        disp_payload = {
            "decision": "HOLD_CONFIRMED_FRAUD",
            "analyst_id": "ANALYST-JANE-44",
            "rationale": "Confirmed suspicious bank modification and near-threshold splitting.",
        }
        disp_resp = await client.post(f"/api/v1/cases/{case_id}/disposition", json=disp_payload)
        assert disp_resp.status_code == 200
        disp_data = disp_resp.json()
        assert disp_data["status"] == "DISPOSED"
        assert disp_data["analyst_decision"] == "HOLD_CONFIRMED_FRAUD"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_monitoring_metrics_and_drift_endpoints():
    """GET /api/v1/monitoring/metrics and /drift should return operational reports."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        m_resp = await client.get("/api/v1/monitoring/metrics")
        assert m_resp.status_code == 200
        metrics = m_resp.json()
        assert "business_metrics" in metrics
        assert "system_telemetry" in metrics
        assert "llmops_metrics" in metrics

        d_resp = await client.get("/api/v1/monitoring/drift")
        assert d_resp.status_code == 200
        drift = d_resp.json()
        assert "overall_severity" in drift
        assert "max_psi" in drift
