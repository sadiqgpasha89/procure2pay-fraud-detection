"""audit-service — the only writer of the forensic ledger (a hash chain needs exactly one writer)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from p2p.common.config import get_settings
from p2p.common.ops import BearerAuthMiddleware, configure_logging, metrics_endpoint
from p2p.services.audit.ledger import BigQuerySink, FileSink, GcsSink, Ledger


class Entry(BaseModel):
    event_type: str
    record_id: str
    payload: dict[str, Any] = {}
    actor: str = "SYSTEM"


def build_ledger() -> Ledger:
    s = get_settings()
    primary = GcsSink(s.AUDIT_GCS_BUCKET) if s.AUDIT_SINK == "gcs" else FileSink(s.ledger_path)
    mirrors = [BigQuerySink(s.AUDIT_BQ_TABLE)] if s.AUDIT_BQ_TABLE else []
    return Ledger(primary, mirrors)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("audit")
    app.state.ledger = build_ledger()
    yield


app = FastAPI(title="p2p audit-service", version="2.0.0", lifespan=lifespan)
app.add_route("/metrics", metrics_endpoint)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "blocks": len(app.state.ledger.chain)}


@app.post("/v1/entries")
def append(e: Entry) -> dict:
    b = app.state.ledger.append(e.event_type, e.record_id, e.payload, e.actor)
    return {"index": b["index"], "hash": b["hash"]}


@app.get("/v1/verify")
def verify() -> dict:
    ok, why = app.state.ledger.verify()
    return {"valid": ok, "detail": why, "blocks": len(app.state.ledger.chain), "head": app.state.ledger.chain[-1]["hash"] if app.state.ledger.chain else None}


@app.get("/v1/entries")
def entries(limit: int = 50, record_id: str | None = None) -> list[dict]:
    chain = app.state.ledger.chain
    rows = [b for b in chain if record_id is None or b["record_id"] == record_id]
    return rows[-limit:]


@app.get("/v1/entries/{index}")
def entry(index: int) -> dict:
    chain = app.state.ledger.chain
    if not 0 <= index < len(chain):
        raise HTTPException(404, "no such block")
    return chain[index]


app.add_middleware(BearerAuthMiddleware)
