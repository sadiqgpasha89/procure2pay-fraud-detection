"""LLM System Metrics tracking RAR, FHR, context efficiency, and latency."""
from typing import Dict, List
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class LLMOperationalMetrics(BaseModel):
    total_dispatches: int
    accepted_recommendations: int
    overridden_recommendations: int
    recommendation_acceptance_rate: float  # Target > 0.85
    total_hallucinations_detected: int
    factual_hallucination_rate: float     # Target < 0.1% (0.001)
    average_synthesis_latency_sec: float  # Target < 8.0s
    total_tokens_consumed: int


class LLMMetricsTracker:
    """Tracks operational agentic AI and LLM performance metrics."""

    def __init__(self):
        self.dispatches: int = 0
        self.acceptances: int = 0
        self.overrides: int = 0
        self.hallucinations: int = 0
        self.latencies_sec: List[float] = []
        self.tokens_consumed: int = 0

    def record_agent_dispatch(self, latency_sec: float, tokens_estimated: int = 450):
        self.dispatches += 1
        self.latencies_sec.append(latency_sec)
        self.tokens_consumed += tokens_estimated

    def record_human_decision(self, accepted: bool):
        """Record human analyst disposition."""
        if accepted:
            self.acceptances += 1
        else:
            self.overrides += 1

    def record_hallucination(self):
        self.hallucinations += 1

    def get_metrics(self) -> LLMOperationalMetrics:
        rar = self.acceptances / max(1, self.acceptances + self.overrides)
        fhr = self.hallucinations / max(1, self.dispatches)
        avg_lat = sum(self.latencies_sec) / max(1, len(self.latencies_sec))

        return LLMOperationalMetrics(
            total_dispatches=self.dispatches,
            accepted_recommendations=self.acceptances,
            overridden_recommendations=self.overrides,
            recommendation_acceptance_rate=round(rar, 4),
            total_hallucinations_detected=self.hallucinations,
            factual_hallucination_rate=round(fhr, 4),
            average_synthesis_latency_sec=round(avg_lat, 2),
            total_tokens_consumed=self.tokens_consumed,
        )
