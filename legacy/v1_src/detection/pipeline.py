"""End-to-End Tri-Tier Hybrid Detection Pipeline orchestrating Rules, ML, and Agentic AI."""
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel
import structlog
from src.core.circuit_breaker import CircuitBreaker
from src.core.config import get_settings
from src.core.models import (
    CaseDossier,
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    RecommendedAction,
    RiskClassification,
    Tier1Result,
    Tier2Result,
    Vendor,
)
from src.core.telemetry import global_metrics
from src.data_engineering.audit.worm_ledger import ImmutableWORMLedger, get_worm_ledger
from src.data_engineering.feature_store.store import UnifiedFeatureStore, get_feature_store
from src.detection.tier1_rules.engine import Tier1RuleEngine
from src.detection.tier2_ml.scorer import Tier2MLScorer
from src.detection.tier3_agentic.orchestrator import AgenticOrchestrator

logger = structlog.get_logger(__name__)


class DetectionOutcome(BaseModel):
    """Complete decision outcome of the Tri-Tier Hybrid Detection Pipeline."""
    invoice_id: str
    vendor_id: str
    final_action: RecommendedAction
    tier1_result: Tier1Result
    tier2_result: Optional[Tier2Result] = None
    case_dossier: Optional[CaseDossier] = None
    requires_human_signoff: bool = False
    audit_hash: str = ""
    total_latency_ms: float = 0.0


class TriTierDetectionPipeline:
    """Master orchestrator for the Tri-Tier Detection Framework.
    
    Tier 1: Sub-millisecond Deterministic Rule Engine
    Tier 2: Probabilistic ML Scoring & Graph Analysis
    Tier 3: Multi-Agent AI Orchestration & Case Synthesis
    """

    def __init__(
        self,
        rule_engine: Optional[Tier1RuleEngine] = None,
        ml_scorer: Optional[Tier2MLScorer] = None,
        agentic_orchestrator: Optional[AgenticOrchestrator] = None,
        feature_store: Optional[UnifiedFeatureStore] = None,
        worm_ledger: Optional[ImmutableWORMLedger] = None,
        external_api_circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.settings = get_settings()
        self.rule_engine = rule_engine or Tier1RuleEngine()
        self.ml_scorer = ml_scorer or Tier2MLScorer()
        self.agentic_orchestrator = agentic_orchestrator or AgenticOrchestrator()
        self.feature_store = feature_store or get_feature_store()
        self.worm_ledger = worm_ledger or get_worm_ledger()
        self.circuit_breaker = external_api_circuit_breaker or CircuitBreaker(
            name="external_verification_apis",
            failure_threshold=self.settings.CIRCUIT_BREAKER_FAIL_MAX,
            recovery_timeout_sec=self.settings.CIRCUIT_BREAKER_RESET_TIMEOUT_SEC,
            fallback_penalty=self.settings.FALLBACK_RISK_PENALTY,
        )

    def evaluate_transaction(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        approver_id: Optional[str] = None,
    ) -> DetectionOutcome:
        """Run transaction through Tri-Tier detection gates."""
        start_time = time.perf_counter()
        logger.info("pipeline.evaluating_transaction", invoice_id=invoice.invoice_id, vendor_id=invoice.vendor_id)

        # 0. Assemble features in feature store
        features = self.feature_store.assemble_features(invoice, vendor, po)

        # 1. Tier 1: Deterministic Rule-Based System (RBS)
        tier1_result = self.rule_engine.evaluate(
            invoice=invoice,
            vendor=vendor,
            po=po,
            gr=gr,
            context=features,
        )

        # Check for immediate Tier 1 hard violation block
        if tier1_result.is_blocked:
            total_elapsed = (time.perf_counter() - start_time) * 1000.0
            global_metrics.record_latency("pipeline_decision_latency_ms", total_elapsed)
            global_metrics.increment_counter("tier1_blocks_total")

            # Log to WORM audit
            citation = self.worm_ledger.log_entry(
                event_type="TIER1_POLICY_BLOCK",
                record_id=invoice.invoice_id,
                payload={"violations": tier1_result.violations, "amount": invoice.total_amount}
            )

            return DetectionOutcome(
                invoice_id=invoice.invoice_id,
                vendor_id=invoice.vendor_id,
                final_action=RecommendedAction.BLOCK_TRANSACTION,
                tier1_result=tier1_result,
                requires_human_signoff=True,
                audit_hash=citation.verification_hash,
                total_latency_ms=round(total_elapsed, 3),
            )

        # 2. Tier 2: Probabilistic ML Scoring
        fallback_penalty = self.circuit_breaker.get_risk_penalty()
        effective_approver = approver_id or (po.created_by if po else None)
        tier2_result = self.ml_scorer.score(
            features=features,
            vendor_id=invoice.vendor_id,
            approver_id=effective_approver,
            fallback_penalty=fallback_penalty,
        )

        # If Low Risk (Score < tau1) -> Auto-approve instantly
        if tier2_result.classification == RiskClassification.LOW_RISK:
            total_elapsed = (time.perf_counter() - start_time) * 1000.0
            global_metrics.record_latency("pipeline_decision_latency_ms", total_elapsed)
            global_metrics.increment_counter("auto_approvals_total")

            citation = self.worm_ledger.log_entry(
                event_type="AUTO_APPROVAL",
                record_id=invoice.invoice_id,
                payload={"composite_score": tier2_result.composite_score, "amount": invoice.total_amount}
            )

            return DetectionOutcome(
                invoice_id=invoice.invoice_id,
                vendor_id=invoice.vendor_id,
                final_action=RecommendedAction.AUTO_APPROVE,
                tier1_result=tier1_result,
                tier2_result=tier2_result,
                requires_human_signoff=False,
                audit_hash=citation.verification_hash,
                total_latency_ms=round(total_elapsed, 3),
            )

        # 3. Tier 3: Agentic AI Orchestration (Medium / High Risk)
        case_dossier = self.agentic_orchestrator.run_investigation(
            invoice=invoice,
            tier1_result=tier1_result,
            tier2_result=tier2_result,
            vendor=vendor,
            po=po,
            gr=gr,
            approver_id=effective_approver,
        )

        total_elapsed = (time.perf_counter() - start_time) * 1000.0
        global_metrics.record_latency("pipeline_decision_latency_ms", total_elapsed)
        global_metrics.increment_counter("agentic_cases_dispatched_total")

        # Log case dossier to WORM ledger
        citation = self.worm_ledger.log_entry(
            event_type="CASE_DOSSIER_COMPILED",
            record_id=case_dossier.case_id,
            payload=case_dossier.model_dump()
        )

        return DetectionOutcome(
            invoice_id=invoice.invoice_id,
            vendor_id=invoice.vendor_id,
            final_action=case_dossier.recommended_action,
            tier1_result=tier1_result,
            tier2_result=tier2_result,
            case_dossier=case_dossier,
            requires_human_signoff=True,
            audit_hash=citation.verification_hash,
            total_latency_ms=round(total_elapsed, 3),
        )
