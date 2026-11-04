"""Agentic AI Orchestration Engine implementing deterministic investigation state machine."""
import time
from typing import Any, Dict, Optional
import structlog
from src.core.models import (
    CaseDossier,
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    Tier1Result,
    Tier2Result,
    Vendor,
)
from src.detection.tier3_agentic.specialists.graph_link_analyst import GraphLinkAnalystSpecialist
from src.detection.tier3_agentic.specialists.document_auditor import DocumentAuditorSpecialist
from src.detection.tier3_agentic.specialists.risk_synthesizer import RiskSynthesizerSpecialist
from src.detection.tier2_ml.graph_detector import GraphCollusionDetector

logger = structlog.get_logger(__name__)


class AgenticInvestigationState:
    """State context passed between investigative state machine nodes."""

    def __init__(
        self,
        invoice: Invoice,
        tier1_result: Tier1Result,
        tier2_result: Tier2Result,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        approver_id: Optional[str] = None,
    ):
        self.invoice = invoice
        self.tier1_result = tier1_result
        self.tier2_result = tier2_result
        self.vendor = vendor
        self.po = po
        self.gr = gr
        self.approver_id = approver_id or (po.created_by if po else None)

        self.context_data: Dict[str, Any] = {}
        self.graph_analysis: Dict[str, Any] = {}
        self.document_audit: Dict[str, Any] = {}
        self.case_dossier: Optional[CaseDossier] = None
        self.execution_duration_sec: float = 0.0


class AgenticOrchestrator:
    """Deterministic Multi-Agent Investigation State Machine (Section 5.4)."""

    def __init__(self, graph_detector: Optional[GraphCollusionDetector] = None):
        self.graph_detector = graph_detector or GraphCollusionDetector()
        self.graph_specialist = GraphLinkAnalystSpecialist()
        self.doc_specialist = DocumentAuditorSpecialist()
        self.synthesizer = RiskSynthesizerSpecialist()

    def run_investigation(
        self,
        invoice: Invoice,
        tier1_result: Tier1Result,
        tier2_result: Tier2Result,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        approver_id: Optional[str] = None,
    ) -> CaseDossier:
        """Execute full deterministic state machine pipeline and return verified CaseDossier."""
        start_time = time.perf_counter()
        logger.info("agentic.investigation_started", invoice_id=invoice.invoice_id)

        # Step 1: Initialize State Machine
        state = AgenticInvestigationState(
            invoice=invoice,
            tier1_result=tier1_result,
            tier2_result=tier2_result,
            vendor=vendor,
            po=po,
            gr=gr,
            approver_id=approver_id,
        )

        # Step 2: Query Context Engine & Graph Link Analysis
        graph_findings = self.graph_detector.get_findings(
            vendor_id=invoice.vendor_id,
            approver_id=state.approver_id,
        )
        state.graph_analysis = self.graph_specialist.investigate(
            vendor_id=invoice.vendor_id,
            approver_id=state.approver_id or "UNKNOWN",
            graph_findings=graph_findings,
        )

        # Step 3: Execute Document & Contract Auditor Specialist
        state.document_audit = self.doc_specialist.audit(
            invoice=invoice,
            po=po,
            gr=gr,
        )

        # Step 4: Synthesize Investigation into Canonical Schema
        case_dossier = self.synthesizer.synthesize(
            invoice=invoice,
            tier1_result=tier1_result,
            tier2_result=tier2_result,
            graph_findings=graph_findings,
            document_audit=state.document_audit,
            vendor=vendor,
        )
        state.case_dossier = case_dossier

        elapsed = time.perf_counter() - start_time
        state.execution_duration_sec = elapsed
        logger.info(
            "agentic.investigation_completed",
            case_id=case_dossier.case_id,
            action=case_dossier.recommended_action.value,
            duration_sec=round(elapsed, 3)
        )

        return case_dossier
