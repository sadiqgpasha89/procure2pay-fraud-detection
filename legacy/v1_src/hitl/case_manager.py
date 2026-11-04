"""Human-in-the-Loop (HITL) Case Management and Triage Workflow."""
from functools import lru_cache
from typing import Dict, List, Optional
import structlog
from pydantic import BaseModel
from src.core.models import CaseDossier, RecommendedAction
from src.llmops.metrics_tracker import LLMMetricsTracker

logger = structlog.get_logger(__name__)


class CaseDispositionRequest(BaseModel):
    decision: str  # e.g. "APPROVED", "REJECTED_FRAUD", "OVERRIDDEN"
    analyst_id: str
    rationale: str
    signoff_timestamp: Optional[str] = None


class CaseRecord(BaseModel):
    dossier: CaseDossier
    status: str = "PENDING_REVIEW"  # PENDING_REVIEW, DISPOSED
    analyst_decision: Optional[str] = None
    analyst_id: Optional[str] = None
    analyst_rationale: Optional[str] = None


class CaseManager:
    """Enterprise Case Management Portal backend for fraud investigator triage."""

    def __init__(self, llm_metrics: Optional[LLMMetricsTracker] = None):
        self.cases: Dict[str, CaseRecord] = {}
        self.llm_metrics = llm_metrics or LLMMetricsTracker()
        self.prevented_loss_total: float = 0.0

    def register_case(self, dossier: CaseDossier, invoice_amount: float = 0.0) -> None:
        """Add new case dossier to investigator triage queue."""
        self.cases[dossier.case_id] = CaseRecord(dossier=dossier)
        self.llm_metrics.record_agent_dispatch(latency_sec=1.5)
        logger.info("case_manager.case_registered", case_id=dossier.case_id, action=dossier.recommended_action)

    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        return self.cases.get(case_id)

    def list_cases(self, status: Optional[str] = None) -> List[CaseRecord]:
        """List cases with optional status filter."""
        all_cases = list(self.cases.values())
        if status:
            return [c for c in all_cases if c.status == status]
        return all_cases

    def record_disposition(
        self,
        case_id: str,
        disposition: CaseDispositionRequest,
        invoice_amount: float = 0.0
    ) -> CaseRecord:
        """Record formal human investigator disposition and update metrics."""
        record = self.cases.get(case_id)
        if not record:
            raise KeyError(f"Case with ID '{case_id}' not found.")

        record.status = "DISPOSED"
        record.analyst_decision = disposition.decision
        record.analyst_id = disposition.analyst_id
        record.analyst_rationale = disposition.rationale

        # Check if analyst agreed with agent's recommended action
        is_accepted = (
            disposition.decision.upper() in record.dossier.recommended_action.value
            or "REJECT" in disposition.decision.upper() and "HOLD" in record.dossier.recommended_action.value
        )
        self.llm_metrics.record_human_decision(accepted=is_accepted)

        if "REJECT" in disposition.decision.upper() or "HOLD" in disposition.decision.upper():
            self.prevented_loss_total += invoice_amount

        logger.info(
            "case_manager.disposition_recorded",
            case_id=case_id,
            decision=disposition.decision,
            accepted=is_accepted,
            analyst=disposition.analyst_id
        )
        return record


@lru_cache()
def get_case_manager() -> CaseManager:
    return CaseManager()
