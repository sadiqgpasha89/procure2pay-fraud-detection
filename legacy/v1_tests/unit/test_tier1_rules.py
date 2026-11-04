"""Unit tests for Tier 1 Deterministic Rule-Based System (RBS)."""
import pytest
from datetime import datetime, timedelta
from src.core.models import Invoice, RecommendedAction
from src.detection.tier1_rules.engine import Tier1RuleEngine
from src.detection.tier1_rules.rules import (
    ExactDuplicateInvoiceRule,
    BankChange72hFreezeRule,
    SanctionsOFACRule,
    POCeilingToleranceRule,
    SplitInvoicingHardRule,
)


@pytest.mark.unit
def test_clean_transaction_passes_tier1(clean_invoice, clean_vendor, sample_po, sample_gr):
    """Clean transaction should pass Tier 1 with zero violations and sub-millisecond latency."""
    engine = Tier1RuleEngine()
    result = engine.evaluate(
        invoice=clean_invoice,
        vendor=clean_vendor,
        po=sample_po,
        gr=sample_gr,
    )
    assert not result.is_blocked
    assert len(result.violations) == 0
    assert result.latency_ms < 5.0  # Ultra-fast sub-millisecond benchmark


@pytest.mark.unit
def test_exact_duplicate_invoice_rule(clean_invoice, clean_vendor):
    """Submitting the same invoice ID twice should trigger immediate duplication block."""
    rule = ExactDuplicateInvoiceRule()
    rule.register_seen(clean_invoice.vendor_id, clean_invoice.invoice_id)

    violation = rule.evaluate(invoice=clean_invoice, vendor=clean_vendor)
    assert violation is not None
    assert "Duplicate invoice identifier detected" in violation


@pytest.mark.unit
def test_bank_change_72h_freeze_rule(clean_invoice, fresh_bank_vendor):
    """Bank account modified 12h ago must trigger mandatory 72h payment freeze (Edge Case 4)."""
    rule = BankChange72hFreezeRule(freeze_hours=72)
    violation = rule.evaluate(invoice=clean_invoice, vendor=fresh_bank_vendor)
    assert violation is not None
    assert "Mandatory freeze: Settlement bank account modified" in violation


@pytest.mark.unit
def test_sanctions_ofac_rule(clean_invoice, sanctioned_vendor):
    """Sanctioned vendor must trigger statutory transaction block."""
    rule = SanctionsOFACRule()
    violation = rule.evaluate(invoice=clean_invoice, vendor=sanctioned_vendor)
    assert violation is not None
    assert "Statutory block: Vendor VEND-SANCTIONED" in violation


@pytest.mark.unit
def test_po_ceiling_tolerance_rule(clean_invoice, sample_po):
    """Invoice exceeding PO by >10% tolerance ceiling should trigger violation."""
    rule = POCeilingToleranceRule(tolerance_pct=0.10)
    # Inflate invoice amount by 50% ($7,500 vs $5,000)
    inflated_invoice = clean_invoice.model_copy(update={"total_amount": 7500.00})

    violation = rule.evaluate(invoice=inflated_invoice, po=sample_po)
    assert violation is not None
    assert "Hard ceiling exceeded" in violation


@pytest.mark.unit
def test_split_invoicing_hard_rule(clean_invoice):
    """Smurfing context flag should trigger threshold proximity rule."""
    rule = SplitInvoicingHardRule()
    context = {"smurfing_24h_pattern_detected": 1.0}
    violation = rule.evaluate(invoice=clean_invoice, context=context)
    assert violation is not None
    assert "Smurfing pattern detected" in violation


@pytest.mark.unit
def test_tier1_engine_blocks_on_violation(clean_invoice, sanctioned_vendor):
    """Tier 1 engine should flag is_blocked=True and BLOCK_TRANSACTION action."""
    engine = Tier1RuleEngine()
    result = engine.evaluate(invoice=clean_invoice, vendor=sanctioned_vendor)
    assert result.is_blocked
    assert result.action == RecommendedAction.BLOCK_TRANSACTION
    assert any("SANCTIONS_OFAC_HIT" in v for v in result.violations)
