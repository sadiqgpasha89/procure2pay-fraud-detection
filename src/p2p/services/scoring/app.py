"""scoring-service — Tier 1 + Tier 2 (the latency-critical path, target P99 ≤ 850 ms; measured in the e2e report)."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from p2p.common.models import ScoringResult
from p2p.common.ops import REQUEST_LATENCY, BearerAuthMiddleware, configure_logging, metrics_endpoint
from p2p.ml.scorer import Tier2Scorer
from p2p.services.scoring.engine import ScoringEngine


class ScoreRequest(BaseModel):
    invoice_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("scoring")
    app.state.engine = ScoringEngine()
    for iid in app.state.engine.wh.invoice_ids()[-6:]:  # warm model, SQLite pages and thread-local connections before taking traffic
        app.state.engine.score(iid)
    yield


app = FastAPI(title="p2p scoring-service", version="2.0.0", lifespan=lifespan)
app.add_route("/metrics", metrics_endpoint)


@app.get("/health")
def health() -> dict:
    e: ScoringEngine = app.state.engine
    return {"status": "ok", "model_version": e.scorer.b.version, "circuit": e.breaker.state.value}


@app.get("/v1/model")
def model_card() -> dict:
    return Tier2Scorer.model_card()


@app.post("/v1/score", response_model=ScoringResult)
def score(req: ScoreRequest) -> ScoringResult:
    t0 = time.perf_counter()
    res = app.state.engine.score(req.invoice_id)
    REQUEST_LATENCY.labels("scoring", "/v1/score").observe(time.perf_counter() - t0)
    if res is None:
        raise HTTPException(404, f"invoice {req.invoice_id} not found")
    return res


app.add_middleware(BearerAuthMiddleware)
