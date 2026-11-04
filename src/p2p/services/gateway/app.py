"""gateway — the single entry point the ERP / payment run calls.

  POST /v1/evaluate {invoice_id}  →  Tier 1+2 (scoring-service) → decision in well under a second
      APPROVE      → AUTO_APPROVE
      BLOCK        → BLOCK_TRANSACTION (+ case + async investigation)
      INVESTIGATE  → HOLD_PAYMENT pending review (+ case + async investigation)
  Tier 3 (A2A → investigator) runs *off the hot path* and attaches the dossier to the case when ready.
  Idempotent on invoice (UUIDv5 key) so payment-run retries never double-evaluate (spec §6).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from p2p.common.config import get_settings
from p2p.common.models import Action, Decision, InvestigationRequest, Route, ScoringResult
from p2p.common.ops import DECISIONS, REQUEST_LATENCY, BearerAuthMiddleware, configure_logging, http_client, metrics_endpoint
from p2p.services.gateway.investigator_client import InvestigationError, InvestigatorClient

log = structlog.get_logger(__name__)
NAMESPACE = uuid.UUID("6f1c2f1e-0a54-5b1e-9a43-70f2a7d1c001")
DECISION_ACTION = {Route.APPROVE: Action.AUTO_APPROVE, Route.BLOCK: Action.BLOCK_TRANSACTION, Route.INVESTIGATE: Action.HOLD_PAYMENT}


class EvaluateRequest(BaseModel):
    invoice_id: str
    wait_for_investigation: bool = False  # demos/tests; production callers use the async path


class DecisionStore:
    """Idempotency store. SQLite locally; point DECISION_DB_URL at Cloud SQL (PostgreSQL) so every gateway instance shares it."""

    def __init__(self, url: str):
        from sqlalchemy import create_engine, text

        self._text = text
        self.engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}, pool_pre_ping=True)
        with self.engine.begin() as c:
            c.execute(text("CREATE TABLE IF NOT EXISTS decisions (invoice_id VARCHAR(64) PRIMARY KEY, key VARCHAR(64), body TEXT)"))

    def get(self, invoice_id: str) -> Decision | None:
        with self.engine.connect() as c:
            r = c.execute(self._text("SELECT body FROM decisions WHERE invoice_id=:i"), {"i": invoice_id}).first()
        return Decision.model_validate_json(r[0]) if r else None

    def put(self, d: Decision) -> None:
        with self.engine.begin() as c:
            c.execute(self._text("DELETE FROM decisions WHERE invoice_id=:i"), {"i": d.invoice_id})
            c.execute(self._text("INSERT INTO decisions (invoice_id, key, body) VALUES (:i,:k,:b)"), {"i": d.invoice_id, "k": d.idempotency_key, "b": d.model_dump_json()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("gateway")
    s = get_settings()
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.state.store = DecisionStore(os.getenv("DECISION_DB_URL", f"sqlite:///{s.decision_db_path.resolve()}"))
    app.state.investigator = InvestigatorClient()
    app.state.tasks = set()  # strong refs so background investigations aren't garbage-collected
    app.state.client = http_client(timeout=15.0)
    try:
        await asyncio.wait_for(app.state.investigator.warm(), timeout=20)
    except Exception as exc:  # investigator may boot after the gateway; first real call then pays the cold start
        log.warning("investigator_warmup_skipped", error=str(exc))
    yield
    await app.state.client.aclose()


app = FastAPI(title="p2p gateway", version="2.0.0", lifespan=lifespan)
app.add_route("/metrics", metrics_endpoint)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "background_investigations": len(app.state.tasks)}


async def _post(path_url: str, body: dict, retries: int = 2) -> dict:
    last: Exception | None = None
    for i in range(retries + 1):
        try:
            r = await app.state.client.post(path_url, json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            last = exc
            await asyncio.sleep(0.2 * (i + 1))
    raise HTTPException(502, f"upstream call failed: {path_url}: {last}")


async def _audit(event_type: str, record_id: str, payload: dict) -> str:
    try:
        return (await _post(f"{get_settings().AUDIT_URL}/v1/entries", {"event_type": event_type, "record_id": record_id, "payload": payload}))["hash"]
    except HTTPException:
        log.error("audit_unavailable", event_type=event_type, record_id=record_id)  # decision path stays up; alert on this log
        return ""


async def _investigate(case_id: str, sc: ScoringResult):
    """Tier 3 off the hot path: A2A → dossier → case-service → ledger."""
    req = InvestigationRequest(case_id=case_id, invoice_id=sc.invoice_id, vendor_id=sc.vendor_id, amount=sc.amount, po_reference=sc.po_reference, scoring=sc)
    last: Exception | None = None
    for attempt in range(2):
        try:
            dossier = await app.state.investigator.investigate(req)
            dossier = dossier.model_copy(update={"case_id": case_id})
            await _put(f"{get_settings().CASE_URL}/v1/cases/{case_id}/dossier", dossier.model_dump(mode="json", by_alias=True))
            await _audit("CASE_DOSSIER_COMPILED", case_id, {"action": dossier.recommended_action.value, "evidence": len(dossier.evidence_manifest),
                                                          "guardrail_notes": dossier.guardrail_notes, "llm": dossier.llm_provider})
            return dossier
        except (InvestigationError, HTTPException) as exc:
            last = exc
            log.warning("investigation_attempt_failed", case_id=case_id, attempt=attempt, error=str(exc))
            await asyncio.sleep(1.0)
    await _audit("INVESTIGATION_FAILED", case_id, {"error": str(last)})
    return None


async def _put(url: str, body: dict) -> None:
    r = await app.state.client.put(url, json=body)
    r.raise_for_status()


@app.post("/v1/evaluate", response_model=Decision)
async def evaluate(req: EvaluateRequest) -> Decision:
    t0 = time.perf_counter()
    cached = app.state.store.get(req.invoice_id)
    if cached:
        return cached
    s = get_settings()
    r = await app.state.client.post(f"{s.SCORING_URL}/v1/score", json={"invoice_id": req.invoice_id})
    if r.status_code == 404:
        raise HTTPException(404, f"invoice {req.invoice_id} not found")
    r.raise_for_status()
    sc = ScoringResult.model_validate(r.json())
    action = DECISION_ACTION[sc.route]
    key = str(uuid.uuid5(NAMESPACE, req.invoice_id))
    audit_hash = await _audit(f"DECISION_{sc.route.value}", req.invoice_id, {
        "action": action.value, "violations": sc.tier1.violations, "score": sc.tier2.composite_score if sc.tier2 else None,
        "model_version": sc.tier2.model_version if sc.tier2 else None, "idempotency_key": key})
    case_id, dossier = None, None
    if sc.route is not Route.APPROVE:
        case = await _post(f"{s.CASE_URL}/v1/cases", {"invoice_id": sc.invoice_id, "vendor_id": sc.vendor_id, "amount": sc.amount, "scoring": sc.model_dump(mode="json")})
        case_id = case["case_id"]
        task = asyncio.create_task(_investigate(case_id, sc))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        if req.wait_for_investigation:
            dossier = await task
    d = Decision(invoice_id=sc.invoice_id, vendor_id=sc.vendor_id, idempotency_key=key, final_action=action, route=sc.route, scoring=sc,
                 case_id=case_id, dossier=dossier, requires_human_signoff=sc.route is not Route.APPROVE, audit_hash=audit_hash,
                 total_latency_ms=round((time.perf_counter() - t0) * 1000, 2))
    app.state.store.put(d.model_copy(update={"dossier": None}))
    DECISIONS.labels(sc.route.value, action.value).inc()
    REQUEST_LATENCY.labels("gateway", "/v1/evaluate").observe(time.perf_counter() - t0)
    return d


@app.get("/v1/decisions/{invoice_id}", response_model=Decision)
def get_decision(invoice_id: str) -> Decision:
    d = app.state.store.get(invoice_id)
    if not d:
        raise HTTPException(404, "no decision for invoice")
    return d


@app.get("/v1/pending")
def pending() -> dict:
    return {"background_investigations": len(app.state.tasks)}


app.add_middleware(BearerAuthMiddleware)
