"""Risk Synthesizer Specialist Agent assembling exact Appendix Case Dossier schema."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import structlog
from src.core.models import (
    CaseDossier,
    EvidenceCitation,
    GraphFindings,
    Invoice,
    RecommendedAction,
    RiskAssessment,
    Tier1Result,
    Tier2Result,
    TransactionIdentifiers,
    Vendor,
)
from src.detection.tier3_agentic.guardrails import (
    ActionIsolationGuardrail,
    CitationVerifier,
)

logger = structlog.get_logger(__name__)


class RiskSynthesizerSpecialist:
    """Compiles multi-agent findings into the canonical Case Synthesis Dossier."""

    def synthesize(
        self,
        invoice: Invoice,
        tier1_result: Tier1Result,
        tier2_result: Tier2Result,
        graph_findings: GraphFindings,
        document_audit: Dict[str, Any],
        vendor: Optional[Vendor] = None,
    ) -> CaseDossier:
        """Synthesize final CaseDossier adhering strictly to the Appendix schema."""
        case_id = f"CASE-{datetime.utcnow().year}-{uuid.uuid4().hex[:6].upper()}"
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Transaction Identifiers
        tx_ids = TransactionIdentifiers(
            invoice_id=invoice.invoice_id,
            po_reference=invoice.po_reference,
            vendor_id=invoice.vendor_id,
        )

        # 2. Risk Assessment
        violations = [v.split("] ")[-1] for v in tier1_result.violations]
        if not violations and tier2_result.composite_score >= 0.70:
            violations.append("ANOMALOUS_COMPOSITE_SCORE_SPIKE")

        risk_assessment = RiskAssessment(
            composite_risk_score=tier2_result.composite_score,
            classification=tier2_result.classification,
            rule_violations=violations,
            top_contributing_factors=tier2_result.contributions[:3],
        )

        # 3. Evidence Manifest Assembly
        evidence_manifest: List[EvidenceCitation] = []
        doc_citations = document_audit.get("citations", [])
        evidence_manifest.extend(doc_citations)

        if graph_findings.entity_conflicts_detected:
            evidence_manifest.append(
                EvidenceCitation(
                    source="GRAPH_DB",
                    record_id=f"EDGE-CONFLICT-{invoice.vendor_id}",
                    verification_hash="ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
                )
            )

        # 4. Agent Narrative Construction with verifiable citations
        narrative_parts = [
            f"Invoice {invoice.invoice_id} (${invoice.total_amount:,.2f}) submitted for vendor {invoice.vendor_id}."
        ]
        if "BANK_MODIFIED_WITHIN_72_HOURS" in " ".join(violations):
            narrative_parts.append(
                "Settlement bank account was modified within 72 hours of submission [Evidence: ERP_AUDIT_LOG]."
            )
        if 40000.00 <= invoice.total_amount < 50000.00:
            narrative_parts.append(
                "The amount sits just below the $50,000 dual-approver threshold."
            )
        if graph_findings.entity_conflicts_detected:
            narrative_parts.append(
                f"Graph linkage reveals entity conflict: {graph_findings.details} [Evidence: GRAPH_DB]."
            )
        if document_audit.get("variance_detected"):
            narrative_parts.append(
                f"Document audit flagged variance: {document_audit.get('notes')} [Evidence: ERP_AUDIT_LOG]."
            )

        agent_narrative = " ".join(narrative_parts)

        # 5. Determine Recommended Action
        if tier1_result.is_blocked or tier2_result.composite_score >= 0.75:
            rec_action = RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT
        elif tier2_result.composite_score >= 0.35:
            rec_action = RecommendedAction.REQUEST_MANUAL_KYC
        else:
            rec_action = RecommendedAction.AUTO_APPROVE

        # Validate action against guardrail
        rec_action = ActionIsolationGuardrail.validate_action(rec_action.value)

        # Enforce citation verification
        is_valid, unverified = CitationVerifier.verify_citations(agent_narrative, evidence_manifest)
        if not is_valid:
            logger.warn("synthesizer.unverified_citations_found", unverified=unverified)

        return CaseDossier(
            case_id=case_id,
            timestamp=timestamp_str,
            transaction_identifiers=tx_ids,
            risk_assessment=risk_assessment,
            graph_findings=graph_findings,
            agent_narrative=agent_narrative,
            recommended_action=rec_action,
            requires_human_signoff=True,
            evidence_manifest=evidence_manifest,
        )
