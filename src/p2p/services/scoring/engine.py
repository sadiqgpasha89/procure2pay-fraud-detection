"""Scoring engine = feature assembly + Tier 1 rules + Tier 2 models. Framework-free so it can be unit-tested,
embedded in a Beam/Dataflow DoFn, or packaged as a Vertex AI custom prediction container unchanged."""
from __future__ import annotations

import time

import structlog

from p2p.common.config import get_settings
from p2p.common.models import Route, RiskClass, ScoringResult
from p2p.common.ops import CircuitBreaker
from p2p.data.repository import Warehouse, open_warehouse
from p2p.features.builder import FeatureBuilder
from p2p.features.graph import KnowledgeGraph
from p2p.ml.scorer import Tier2Scorer
from p2p.rules.engine import RuleEngine

log = structlog.get_logger(__name__)


class ScoringEngine:
    def __init__(self, wh: Warehouse | None = None, scorer: Tier2Scorer | None = None):
        s = get_settings()
        self.wh = wh or open_warehouse()
        self.builder = FeatureBuilder(self.wh, KnowledgeGraph.from_warehouse(self.wh))
        self.rules = RuleEngine()
        self.scorer = scorer or Tier2Scorer.load()
        self.breaker = CircuitBreaker("external_verification", s.CIRCUIT_FAIL_MAX, s.CIRCUIT_RESET_SEC, s.FALLBACK_RISK_PENALTY)
        self.simulate_outage = False  # test hook for spec edge case 8 (third-party API downtime)

    def _external_bank_verification(self, vendor_id: str) -> bool:
        """Stand-in for a bank-validation / registry API call, protected by the circuit breaker."""
        if self.simulate_outage:
            raise TimeoutError("bank verification API timed out")
        return True

    def score(self, invoice_id: str) -> ScoringResult | None:
        t0 = time.perf_counter()
        tx = self.builder.load(invoice_id)
        if tx is None:
            return None
        self.breaker.call(self._external_bank_verification, tx.vendor.vendor_id, fallback=lambda *_: True)
        t1 = self.rules.evaluate(tx)
        t2 = self.scorer.score(tx.features, fallback_penalty=self.breaker.risk_penalty())
        if t1.is_blocked:
            route = Route.BLOCK
        elif t2.classification is RiskClass.LOW:
            route = Route.APPROVE
        else:
            route = Route.INVESTIGATE
        return ScoringResult(invoice_id=invoice_id, vendor_id=tx.vendor.vendor_id, amount=tx.invoice.total_amount, po_reference=tx.invoice.po_id, route=route, tier1=t1, tier2=t2, features=tx.features,
                             total_latency_ms=round((time.perf_counter() - t0) * 1000, 3))
