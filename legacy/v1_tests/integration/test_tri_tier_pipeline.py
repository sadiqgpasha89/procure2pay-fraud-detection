"""Integration tests for Tri-Tier Detection Pipeline (Tier 1 -> Tier 2 -> Tier 3)."""
import pytest
from datetime import datetime, timedelta
from src.core.models import Invoice, RecommendedAction, RiskClassification
from src.detection.pipeline import TriTierDetectionPipeline


@pytest.mark.integration
def test_clean_transaction_auto_approval_flow(tri_tier_pipeline, clean_invoice, clean_vendor, sample_po, sample_gr):
    """Clean transaction should navigate Tier 1 & Tier 2 and be instantly Auto-Approved."""
    outcome = tri_tier_pipeline.evaluate_transaction(
        invoice=clean_invoice,
        vendor=clean_vendor,
        po=sample_po,
        gr=sample_gr,
    )
    assert outcome.final_action == RecommendedAction.AUTO_APPROVE
    assert not outcome.requires_human_signoff
    assert outcome.tier1_result.is_blocked is False
    assert outcome.tier2_result is not None
    assert outcome.tier2_result.classification == RiskClassification.LOW_RISK
    assert outcome.case_dossier is None
    assert outcome.audit_hash != ""
    assert outcome.total_latency_ms < 100.0  # Real-time sub-second execution


@pytest.mark.integration
def test_hard_violation_immediate_block_flow(tri_tier_pipeline, clean_invoice, fresh_bank_vendor, sample_po):
    """Bank account modified <72h must halt immediately at Tier 1 with BLOCK_TRANSACTION."""
    outcome = tri_tier_pipeline.evaluate_transaction(
        invoice=clean_invoice,
        vendor=fresh_bank_vendor,
        po=sample_po,
    )
    assert outcome.final_action == RecommendedAction.BLOCK_TRANSACTION
    assert outcome.requires_human_signoff is True
    assert outcome.tier1_result.is_blocked is True
    assert any("BANK_MODIFIED_WITHIN_72_HOURS" in v for v in outcome.tier1_result.violations)
    assert outcome.audit_hash != ""


@pytest.mark.integration
def test_high_risk_triggers_tier3_agentic_investigation(tri_tier_pipeline, clean_vendor, sample_po):
    """High risk transaction passing Tier 1 must trigger Tier 3 Agentic Dossier compilation."""
    # Create an invoice that appears clean to deterministic rules but highly anomalous in features:
    # 4x 90-day average amount, near $50k dual approver limit ($49,850), submitted off hours
    suspicious_invoice = Invoice(
        invoice_id="INV-SUSPICIOUS-99",
        vendor_id=clean_vendor.vendor_id,
        po_reference=sample_po.po_id,
        net_amount=49850.00,
        total_amount=49850.00,
        bank_iban=clean_vendor.bank_iban,
        invoice_date=datetime(2026, 9, 13, 23, 30, 0),  # Sunday midnight off-hours
    )

    # Set historical profile where normal spend is $5,000 to induce a 10x spike anomaly
    from src.data_engineering.feature_store.behavioral_features import HistoricalSpendProfile
    history = HistoricalSpendProfile(
        vendor_id=clean_vendor.vendor_id,
        amounts_90d=[4800.0, 5200.0, 5000.0, 4900.0, 5100.0]
    )
    tri_tier_pipeline.feature_store.set_historical_profile(clean_vendor.vendor_id, history)

    matching_po = sample_po.model_copy(update={"amount": 50000.00})
    outcome = tri_tier_pipeline.evaluate_transaction(
        invoice=suspicious_invoice,
        vendor=clean_vendor,
        po=matching_po,
    )

    assert outcome.tier1_result.is_blocked is False
    assert outcome.tier2_result is not None
    assert outcome.tier2_result.composite_score >= 0.35  # Medium/High risk
    assert outcome.requires_human_signoff is True
    assert outcome.case_dossier is not None
    assert outcome.case_dossier.case_id.startswith("CASE-")
    assert len(outcome.case_dossier.evidence_manifest) > 0
    assert outcome.audit_hash != ""
