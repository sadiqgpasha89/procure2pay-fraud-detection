"""Integration test suite explicitly covering all 8 scenarios from Section 9 Edge Case Matrix."""
from datetime import datetime, timedelta
import pytest
from src.core.circuit_breaker import CircuitBreaker, CircuitState
from src.core.models import Invoice, RecommendedAction, Vendor
from src.data_engineering.ingestion.entity_resolution import EntityResolver
from src.data_engineering.ingestion.ocr_extractor import OCRExtractor
from src.detection.tier1_rules.engine import Tier1RuleEngine
from src.detection.tier2_ml.unsupervised_detector import UnsupervisedAnomalyDetector
from src.detection.pipeline import TriTierDetectionPipeline


@pytest.mark.edge_case
def test_edge_case_1_sparse_label_cold_start():
    """Edge Case 1: Sparse Label Cold-Start.
    
    Mitigation: Unsupervised anomaly detector (Isolation Forest + Autoencoder) detects
    anomalies without requiring historic labeled fraud examples.
    """
    detector = UnsupervisedAnomalyDetector()
    novel_vector = {
        "po_amount_variance_ratio": 0.90,
        "vendor_bank_account_age_days": 0.5,
        "extraction_confidence": 0.30,
        "vendor_zscore_amount_last_90d": 6.2,
        "invoice_amount_vs_90d_avg_ratio": 7.5,
    }
    score = detector.score_anomaly(novel_vector)
    assert score >= 0.50  # Highly anomalous score without needing supervised training labels


@pytest.mark.edge_case
def test_edge_case_2_high_ocr_error_rates():
    """Edge Case 2: High OCR Error Rates.
    
    Mitigation: Attach field-level confidence Ce. Documents with Ce < 0.85 route to
    async validation queue before score generation.
    """
    extractor = OCRExtractor(confidence_threshold=0.85)
    unclear_scan_text = "Inv..??? amt: $$$.???"
    res = extractor.process_raw_document(unclear_scan_text, base_confidence=0.40)

    assert res.overall_confidence < 0.85
    assert res.routed_to_validation_queue is True
    assert res in extractor.async_validation_queue


@pytest.mark.edge_case
def test_edge_case_3_supplier_name_collisions():
    """Edge Case 3: Supplier Name Collisions.
    
    Mitigation: Multi-attribute resolution requiring intersection of Tax ID, registration jurisdiction,
    bank routing, and physical address before establishing graph edges.
    """
    legit_corp = Vendor(
        vendor_id="VEND-LEGIT-CORP",
        legal_name="Consolidated Logistics Inc",
        tax_id="88-1122334",
        bank_iban="US88LEGITBANK",
        address="100 Logistics Blvd, Atlanta, GA",
    )
    collision_imposter = Vendor(
        vendor_id="VEND-IMPOSTER-CORP",
        legal_name="Consolidated Logistics LLC",  # Near identical name
        tax_id="00-9988776",                    # Different Tax ID
        bank_iban="US00FRAUDBANK",               # Different Bank
        address="999 Fake Rd, Wilmington, DE",
    )
    resolver = EntityResolver(existing_vendors=[legit_corp])
    matches = resolver.find_matches(collision_imposter)

    assert len(matches) > 0
    assert matches[0].is_name_collision_imposter is True
    assert matches[0].name_similarity >= 0.85
    assert matches[0].tax_id_exact_match is False


@pytest.mark.edge_case
def test_edge_case_4_mid_cycle_bank_modification(clean_invoice, fresh_bank_vendor):
    """Edge Case 4: Mid-Cycle Bank Modification.
    
    Mitigation: Strict policy rule freezes invoice execution for 72 hours upon bank details
    modification, mandating out-of-band dual-authorization.
    """
    engine = Tier1RuleEngine()
    result = engine.evaluate(invoice=clean_invoice, vendor=fresh_bank_vendor)

    assert result.is_blocked is True
    assert result.action == RecommendedAction.BLOCK_TRANSACTION
    assert any("BANK_MODIFIED_WITHIN_72_HOURS" in v for v in result.violations)


@pytest.mark.edge_case
def test_edge_case_5_collusion_across_actors(tri_tier_pipeline, clean_invoice, graph_engine):
    """Edge Case 5: Collusion Across Actors.
    
    Mitigation: Knowledge graph link-analysis traverses corporate registry filings and
    shared residential address between AP clerk and vendor director.
    """
    # Vendor and approver share Evergreen Terrace residential address
    collusive_inv = clean_invoice.model_copy(update={"vendor_id": "VEND-COLLUSIVE"})
    findings = graph_engine.analyze_entity_conflicts(
        vendor_id="VEND-COLLUSIVE",
        approver_id="EMP-APPROVER-99"
    )
    assert findings.entity_conflicts_detected is True
    assert "742 evergreen" in findings.details.lower()


@pytest.mark.edge_case
def test_edge_case_6_smurfing_split_invoicing(tri_tier_pipeline, clean_invoice, clean_vendor, sample_po):
    """Edge Case 6: Smurfing (Split Invoicing).
    
    Mitigation: Rolling window aggregation computing cumulative spend grouped by (Vendor_ID, Cost_Center)
    over 24-hour, 7-day, and 30-day windows.
    """
    # Record previous spend of $45,000 within last 6 hours
    tri_tier_pipeline.feature_store.record_transaction_spend(
        vendor_id=clean_vendor.vendor_id,
        invoice_id="INV-SMURF-1",
        amount=45000.00,
        cost_center=sample_po.cost_center,
        timestamp=datetime.utcnow() - timedelta(hours=6),
    )

    # Submit second invoice of $48,000 (each invoice is < $50,000, but 24h sum = $93,000!)
    second_inv = clean_invoice.model_copy(update={
        "invoice_id": "INV-SMURF-2",
        "total_amount": 48000.00,
    })

    feats = tri_tier_pipeline.feature_store.assemble_features(second_inv, clean_vendor, sample_po)
    assert feats["smurfing_24h_pattern_detected"] == 1.0
    assert feats["rolling_24h_spend_vs_approval_limit"] >= 0.90


@pytest.mark.edge_case
def test_edge_case_7_novel_zero_day_fraud_vectors():
    """Edge Case 7: Novel / Zero-Day Fraud Vectors.
    
    Mitigation: Unsupervised autoencoder flags high reconstruction error and routes to investigator triage.
    """
    detector = UnsupervisedAnomalyDetector()
    novel_unseen_attack = {
        "po_amount_variance_ratio": 0.88,
        "vendor_bank_account_age_days": 0.2,
        "extraction_confidence": 0.25,
        "is_submitted_off_hours": 1.0,
        "vendor_zscore_amount_last_90d": 7.0,
        "invoice_amount_vs_90d_avg_ratio": 8.0,
    }
    score = detector.score_anomaly(novel_unseen_attack)
    assert score >= 0.50


@pytest.mark.edge_case
def test_edge_case_8_third_party_api_downtime():
    """Edge Case 8: Third-Party API Downtime.
    
    Mitigation: Circuit breaker falls back to cached verifications, logs a degradation event,
    and assigns a fallback penalty score (+0.15).
    """
    cb = CircuitBreaker("sanctions_service", failure_threshold=2, recovery_timeout_sec=5.0, fallback_penalty=0.15)

    def flaky_sanctions_call():
        raise ConnectionResetError("Third-party gateway timeout")

    # Trip circuit breaker
    for _ in range(2):
        cb.call(flaky_sanctions_call, fallback=lambda: {"cached": True})

    assert cb.state == CircuitState.OPEN
    penalty = cb.get_risk_penalty()
    assert penalty == 0.15  # Fallback risk penalty applied to transaction score
