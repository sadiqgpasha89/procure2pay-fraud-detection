"""Unit tests for Tier 3 Agentic AI Orchestrator, Specialists, and Guardrails."""
import pytest
from src.core.models import (
    EvidenceCitation,
    GraphFindings,
    RecommendedAction,
    RiskClassification,
    Tier1Result,
    Tier2Result,
)
from src.detection.tier3_agentic.guardrails import (
    ActionIsolationGuardrail,
    CitationVerifier,
    PIIMasker,
)
from src.detection.tier3_agentic.orchestrator import AgenticOrchestrator
from src.detection.tier3_agentic.specialists.document_auditor import DocumentAuditorSpecialist
from src.detection.tier3_agentic.specialists.graph_link_analyst import GraphLinkAnalystSpecialist
from src.detection.tier3_agentic.specialists.risk_synthesizer import RiskSynthesizerSpecialist


@pytest.mark.unit
def test_pii_masker():
    """PII Masker should redact Tax IDs, SSNs, and IBANs."""
    raw = "Vendor Tax ID 12-3456789 and IBAN US89BANK00001234567890."
    masked = PIIMasker.mask_text(raw)
    assert "12-3456789" not in masked
    assert "US89BANK00001234567890" not in masked


@pytest.mark.unit
def test_citation_verifier():
    """CitationVerifier should extract [Evidence: ...] tokens and verify against manifest."""
    narrative = (
        "Invoice submitted following bank change [Evidence: ERP_AUDIT_LOG]. "
        "Director shared address with approver [Evidence: GRAPH_DB]."
    )
    manifest = [
        EvidenceCitation(source="ERP_AUDIT_LOG", record_id="REC-1", verification_hash="hash1"),
        EvidenceCitation(source="GRAPH_DB", record_id="EDGE-1", verification_hash="hash2"),
    ]
    is_valid, unverified = CitationVerifier.verify_citations(narrative, manifest)
    assert is_valid
    assert len(unverified) == 0

    invalid_narrative = narrative + " Ghost delivery claimed [Evidence: FABRICATED_SOURCE]."
    is_valid_bad, unverified_bad = CitationVerifier.verify_citations(invalid_narrative, manifest)
    assert not is_valid_bad
    assert len(unverified_bad) > 0


@pytest.mark.unit
def test_action_isolation_guardrail():
    """ActionIsolationGuardrail must enforce strongly-typed allowed actions."""
    valid_action = ActionIsolationGuardrail.validate_action("HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT")
    assert valid_action == RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT

    # Invalid arbitrary string coerced to safe default
    fallback = ActionIsolationGuardrail.validate_action("DELETE_VENDOR_IMMEDIATELY")
    assert fallback == RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT


@pytest.mark.unit
def test_document_auditor_variance_detection(clean_invoice, sample_po, sample_gr):
    """Document auditor should identify discrepancies between invoice, PO, and GR."""
    auditor = DocumentAuditorSpecialist()
    # Modify invoice amount to create variance
    varied_invoice = clean_invoice.model_copy(update={"total_amount": 5250.00})

    result = auditor.audit(varied_invoice, po=sample_po, gr=sample_gr)
    assert result["variance_detected"]
    assert "differs from PO" in result["notes"]
    assert len(result["citations"]) > 0


@pytest.mark.unit
def test_risk_synthesizer_appendix_schema(clean_invoice, clean_vendor):
    """Synthesizer must produce CaseDossier matching exact fields of Appendix specification."""
    synthesizer = RiskSynthesizerSpecialist()

    tier1 = Tier1Result(is_blocked=False, violations=["BANK_MODIFIED_WITHIN_72_HOURS"])
    tier2 = Tier2Result(composite_score=0.892, classification=RiskClassification.HIGH_RISK)
    graph_findings = GraphFindings(
        entity_conflicts_detected=True,
        details="Shared residential address found between Vendor Director and AP Approver [Node ID: EMP-10492]."
    )
    doc_audit = {"variance_detected": False, "notes": "No variance.", "citations": []}

    dossier = synthesizer.synthesize(
        invoice=clean_invoice,
        tier1_result=tier1,
        tier2_result=tier2,
        graph_findings=graph_findings,
        document_audit=doc_audit,
        vendor=clean_vendor,
    )

    # Validate all Appendix schema fields exist and match format
    assert dossier.case_id.startswith("CASE-")
    assert dossier.risk_assessment.composite_risk_score == 0.892
    assert dossier.risk_assessment.classification == RiskClassification.HIGH_RISK
    assert "BANK_MODIFIED_WITHIN_72_HOURS" in dossier.risk_assessment.rule_violations
    assert dossier.graph_findings.entity_conflicts_detected
    assert dossier.recommended_action == RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT
    assert dossier.requires_human_signoff is True
    assert len(dossier.evidence_manifest) > 0


@pytest.mark.unit
def test_agentic_orchestrator_end_to_end(clean_invoice, clean_vendor, sample_po, graph_engine):
    """Agentic Orchestrator should run the full 5-step deterministic state machine."""
    orchestrator = AgenticOrchestrator()
    tier1 = Tier1Result(is_blocked=False)
    tier2 = Tier2Result(composite_score=0.78, classification=RiskClassification.HIGH_RISK)

    dossier = orchestrator.run_investigation(
        invoice=clean_invoice,
        tier1_result=tier1,
        tier2_result=tier2,
        vendor=clean_vendor,
        po=sample_po,
    )
    assert dossier is not None
    assert dossier.case_id.startswith("CASE-")
    assert dossier.recommended_action in list(RecommendedAction)
