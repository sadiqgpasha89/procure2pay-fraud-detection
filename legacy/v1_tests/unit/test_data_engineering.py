"""Unit tests for Data Engineering: Ingestion, Entity Resolution, Feature Store, and WORM Audit."""
import pytest
from datetime import datetime, timedelta
from src.core.models import Invoice, Vendor
from src.data_engineering.audit.worm_ledger import ImmutableWORMLedger
from src.data_engineering.feature_store.behavioral_features import (
    HistoricalSpendProfile,
    compute_behavioral_features,
)
from src.data_engineering.feature_store.real_time_features import compute_real_time_features
from src.data_engineering.feature_store.store import UnifiedFeatureStore
from src.data_engineering.ingestion.entity_resolution import EntityResolver
from src.data_engineering.ingestion.erp_adapters import ERPAdapterFactory
from src.data_engineering.ingestion.ocr_extractor import OCRExtractor


@pytest.mark.unit
def test_ocr_extraction_confidence_routing():
    """Documents with Ce < 0.85 must route to async validation queue (Edge Case 2)."""
    extractor = OCRExtractor(confidence_threshold=0.85)

    # Clean high quality document
    clean_text = """
    Invoice: INV-990214
    Vendor ID: VEND-8821
    Tax ID: 12-3456789
    PO: PO-554102
    Total Amount: $49,850.00
    IBAN: US89BANK00001234567890
    Item 1: Consulting Services Qty: 1 Price: 49850.00
    """
    res_clean = extractor.process_raw_document(clean_text, base_confidence=0.95)
    assert res_clean.overall_confidence >= 0.85
    assert not res_clean.routed_to_validation_queue

    # Degraded, low confidence document
    blurry_text = "Inv??? total ???"
    res_blurry = extractor.process_raw_document(blurry_text, base_confidence=0.50)
    assert res_blurry.overall_confidence < 0.85
    assert res_blurry.routed_to_validation_queue
    assert res_blurry in extractor.async_validation_queue


@pytest.mark.unit
def test_entity_resolution_name_collision_imposter():
    """Supplier with near identical name but different Tax ID / Bank is a collision imposter (Edge Case 3)."""
    legit_vendor = Vendor(
        vendor_id="VEND-LEGIT",
        legal_name="Apex Global Technologies Inc",
        tax_id="11-2233445",
        bank_iban="US11LEGITBANK001",
        address="100 Technology Plaza, San Jose, CA",
    )
    imposter_vendor = Vendor(
        vendor_id="VEND-IMPOSTER",
        legal_name="Apex Global Technologies LLC",  # Near identical name!
        tax_id="99-8877665",                       # Different Tax ID!
        bank_iban="US99FRAUDBANK999",              # Different Bank!
        address="999 Shadow Alley, Wilmington, DE",
    )

    resolver = EntityResolver(existing_vendors=[legit_vendor])
    matches = resolver.find_matches(imposter_vendor)

    assert len(matches) > 0
    match = matches[0]
    assert match.is_name_collision_imposter is True
    assert match.tax_id_exact_match is False
    assert match.bank_iban_exact_match is False


@pytest.mark.unit
def test_erp_adapters_sap_and_netsuite():
    """ERP adapters should correctly map SAP and NetSuite structures into canonical Invoice."""
    # SAP IDoc payload
    sap_payload = {
        "BELNR": "SAP-INV-001",
        "LIFNR": "VEND-SAP",
        "EBELN": "SAP-PO-100",
        "WRBTR": 12500.00,
        "WMWST": 500.00,
        "WAERS": "USD",
        "IBAN": "US88SAP0001",
        "ITEMS": [{"TXZ01": "Server Hardware", "MENGE": 1, "NETPR": 12000.0, "NETWR": 12000.0}],
    }
    sap_inv = ERPAdapterFactory.parse("SAP", sap_payload)
    assert sap_inv.invoice_id == "SAP-INV-001"
    assert sap_inv.total_amount == 12500.00
    assert sap_inv.net_amount == 12000.00

    # NetSuite payload
    ns_payload = {
        "tranId": "NS-INV-999",
        "entity": {"id": "VEND-NS"},
        "createdFrom": {"id": "NS-PO-555"},
        "total": 8400.00,
        "subtotal": 8000.00,
        "currencyName": "USD",
    }
    ns_inv = ERPAdapterFactory.parse("NetSuite", ns_payload)
    assert ns_inv.invoice_id == "NS-INV-999"
    assert ns_inv.total_amount == 8400.00


@pytest.mark.unit
def test_worm_ledger_integrity_and_hash_chain():
    """WORM ledger must cryptographically chain entries and detect any tampering."""
    ledger = ImmutableWORMLedger()

    cit1 = ledger.log_entry("INVOICE_INGESTED", "INV-001", {"amount": 5000.0})
    cit2 = ledger.log_entry("TIER1_EVALUATION", "INV-001", {"status": "CLEAN"})
    cit3 = ledger.log_entry("PAYMENT_RELEASED", "INV-001", {"batch": "BATCH-1"})

    assert len(ledger.chain) == 4  # Genesis + 3 entries
    assert ledger.verify_integrity() is True

    # Tamper with an entry in the middle of the chain
    ledger.chain[2].payload["status"] = "TAMPERED_FRAUD"
    assert ledger.verify_integrity() is False
