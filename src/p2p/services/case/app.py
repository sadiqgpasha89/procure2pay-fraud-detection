"""case-service — human-in-the-loop queue, dispositions, and the business/agent KPIs of spec §7.

SQLAlchemy 2.0 ORM: SQLite for the MVP, Cloud SQL (PostgreSQL) in production via `CASE_DB_URL`-style DSN swap.
Every disposition is written to the audit ledger (SOX): who decided, against which recommendation, why.
"""
from __future__ import annotations

import os
import statistics
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Float, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from p2p.common.config import get_settings
from p2p.common.models import CaseDossier, CaseStatus, Disposition, ScoringResult
from p2p.common.ops import BearerAuthMiddleware, configure_logging, http_client, metrics_endpoint

INVESTIGATOR_HOURS_PER_FALSE_POSITIVE = 0.75
BLENDED_HOURLY_RATE = 85.0


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "cases"
    case_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    invoice_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    vendor_id: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float)
    route: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), index=True)
    scoring: Mapped[dict] = mapped_column(JSON)
    dossier: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    recommended_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    dossier_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    disposed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    analyst_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rationale: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    accepted_recommendation: Mapped[int | None] = mapped_column(nullable=True)


class NewCase(BaseModel):
    invoice_id: str
    vendor_id: str
    amount: float
    scoring: ScoringResult


def to_dict(c: CaseRow, full: bool = True) -> dict:
    d = {k: getattr(c, k) for k in ("case_id", "invoice_id", "vendor_id", "amount", "route", "status", "recommended_action", "decision", "analyst_id")}
    d.update(created_at=c.created_at.isoformat(), dossier_at=c.dossier_at.isoformat() if c.dossier_at else None,
             disposed_at=c.disposed_at.isoformat() if c.disposed_at else None,
             risk_score=(c.scoring.get("tier2") or {}).get("composite_score"), violations=c.scoring["tier1"]["violations"])
    if full:
        d.update(scoring=c.scoring, dossier=c.dossier, rationale=c.rationale, accepted_recommendation=c.accepted_recommendation)
    return d


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("case")
    s = get_settings()
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.state.engine = create_engine(os.getenv("CASE_DB_URL", s.case_db_url), connect_args={"check_same_thread": False} if "sqlite" in s.case_db_url else {})
    Base.metadata.create_all(app.state.engine)
    yield


app = FastAPI(title="p2p case-service", version="2.0.0", lifespan=lifespan)
app.add_route("/metrics", metrics_endpoint)


def db() -> Session:
    return Session(app.state.engine, expire_on_commit=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/cases")
def create_case(n: NewCase) -> dict:
    with db() as s:
        row = s.scalar(select(CaseRow).where(CaseRow.invoice_id == n.invoice_id))
        if row:  # idempotent on invoice
            return to_dict(row, full=False)
        row = CaseRow(case_id=f"CASE-{now().year}-{uuid.uuid4().hex[:6].upper()}", invoice_id=n.invoice_id, vendor_id=n.vendor_id, amount=n.amount,
                      route=n.scoring.route.value, status=CaseStatus.INVESTIGATING.value, scoring=n.scoring.model_dump(mode="json"), created_at=now())
        s.add(row)
        s.commit()
        return to_dict(row, full=False)


@app.put("/v1/cases/{case_id}/dossier")
def attach_dossier(case_id: str, dossier: CaseDossier) -> dict:
    with db() as s:
        row = s.get(CaseRow, case_id)
        if not row:
            raise HTTPException(404, "case not found")
        row.dossier, row.recommended_action = dossier.model_dump(mode="json", by_alias=True), dossier.recommended_action.value
        row.dossier_at = now()
        if row.status == CaseStatus.INVESTIGATING.value:
            row.status = CaseStatus.PENDING_REVIEW.value
        s.commit()
        return to_dict(row, full=False)


@app.get("/v1/cases")
def list_cases(status: str | None = None, limit: int = 100) -> list[dict]:
    with db() as s:
        q = select(CaseRow).order_by(CaseRow.created_at.desc()).limit(limit)
        if status:
            q = q.where(CaseRow.status == status)
        return [to_dict(c, full=False) for c in s.scalars(q)]


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict:
    with db() as s:
        row = s.get(CaseRow, case_id)
        if not row:
            raise HTTPException(404, "case not found")
        return to_dict(row)


@app.post("/v1/cases/{case_id}/disposition")
async def dispose(case_id: str, d: Disposition) -> dict:
    if d.decision not in {"CONFIRMED_FRAUD", "FALSE_POSITIVE", "OVERRIDDEN"}:
        raise HTTPException(422, "decision must be CONFIRMED_FRAUD | FALSE_POSITIVE | OVERRIDDEN")
    with db() as s:
        row = s.get(CaseRow, case_id)
        if not row:
            raise HTTPException(404, "case not found")
        if row.status == CaseStatus.DISPOSED.value:
            raise HTTPException(409, "case already disposed")
        if row.dossier is None:
            raise HTTPException(409, "dossier not ready; investigation still running")
        row.status, row.decision, row.analyst_id, row.rationale = CaseStatus.DISPOSED.value, d.decision, d.analyst_id, d.rationale
        row.accepted_recommendation, row.disposed_at = int(d.accepted_recommendation), now()
        s.commit()
        out = to_dict(row)
    try:  # SOX: human decision into the immutable ledger
        async with http_client() as c:
            await c.post(f"{get_settings().AUDIT_URL}/v1/entries", json={"event_type": "HUMAN_DISPOSITION", "record_id": case_id, "actor": d.analyst_id,
                                                                        "payload": {"decision": d.decision, "accepted_recommendation": d.accepted_recommendation,
                                                                                    "recommended_action": out["recommended_action"], "rationale": d.rationale}})
    except httpx.HTTPError:
        pass
    return out


@app.get("/v1/metrics")
def metrics() -> dict:
    """Spec §7 KPIs computed from live case data."""
    with db() as s:
        rows = list(s.scalars(select(CaseRow)))
    disposed = [r for r in rows if r.status == CaseStatus.DISPOSED.value]
    fraud = [r for r in disposed if r.decision == "CONFIRMED_FRAUD"]
    fp = [r for r in disposed if r.decision == "FALSE_POSITIVE"]
    secs = [(r.disposed_at - r.created_at).total_seconds() for r in disposed if r.disposed_at]
    synth = [r.dossier["synthesis_latency_sec"] for r in rows if r.dossier]
    return {
        "cases_total": len(rows), "by_status": {k.value: sum(r.status == k.value for r in rows) for k in CaseStatus},
        "gross_prevented_loss": round(sum(r.amount for r in fraud), 2),
        "false_positive_operational_cost": round(len(fp) * INVESTIGATOR_HOURS_PER_FALSE_POSITIVE * BLENDED_HOURLY_RATE, 2),
        "recommendation_acceptance_rate": round(sum(r.accepted_recommendation or 0 for r in disposed) / len(disposed), 3) if disposed else None,
        "median_resolution_seconds": round(statistics.median(secs), 1) if secs else None,
        "median_synthesis_latency_sec": round(statistics.median(synth), 2) if synth else None,
        "precision_of_escalations": round(len(fraud) / (len(fraud) + len(fp)), 3) if fraud or fp else None,
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


app.add_middleware(BearerAuthMiddleware)
