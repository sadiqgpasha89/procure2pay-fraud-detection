"""Shared pytest fixtures and test data generators for Procure-to-Pay fraud tests."""
from datetime import datetime, timedelta
import pytest
from src.core.models import (
    GoodsReceipt,
    Invoice,
    InvoiceLineItem,
    PurchaseOrder,
    Vendor,
)
from src.data_engineering.feature_store.graph_features import KnowledgeGraphEngine
from src.data_engineering.feature_store.store import UnifiedFeatureStore
from src.detection.tier1_rules.engine import Tier1RuleEngine
from src.detection.tier2_ml.scorer import Tier2MLScorer
from src.detection.tier3_agentic.orchestrator import AgenticOrchestrator
from src.detection.pipeline import TriTierDetectionPipeline


@pytest.fixture
def clean_vendor() -> Vendor:
    """Mature, legitimate supplier."""
    return Vendor(
        vendor_id="VEND-001",
        legal_name="Global Office Supplies Inc",
        tax_id="12-3456789",
        bank_iban="US89BANK00001234567890",
        bank_account_updated_at=datetime.utcnow() - timedelta(days=400),
        registration_date=datetime.utcnow() - timedelta(days=800),
        address="100 Enterprise Way, Dallas, TX 75001",
        phone="+1-214-555-0199",
        email_domain="globaloffice.com",
        directors=["Jane Doe", "Robert Smith"],
        is_sanctioned=False,
    )


@pytest.fixture
def fresh_bank_vendor() -> Vendor:
    """Vendor whose settlement bank account was modified 12 hours ago (Edge Case 4)."""
    return Vendor(
        vendor_id="VEND-002",
        legal_name="Apex Logistics LLC",
        tax_id="98-7654321",
        bank_iban="US44DIVERTBANK9988776655",
        bank_account_updated_at=datetime.utcnow() - timedelta(hours=12),
        registration_date=datetime.utcnow() - timedelta(days=200),
        address="500 Transit Rd, Chicago, IL 60601",
        phone="+1-312-555-0144",
        email_domain="apexlogistics.com",
        directors=["Victor Divert"],
        is_sanctioned=False,
    )


@pytest.fixture
def sanctioned_vendor() -> Vendor:
    """Vendor flagged on sanctions list."""
    return Vendor(
        vendor_id="VEND-SANCTIONED",
        legal_name="Blacklisted Maritime Corp",
        tax_id="00-0000000",
        bank_iban="RU00SECRET00000000",
        bank_account_updated_at=datetime.utcnow() - timedelta(days=100),
        address="1 Offshore Pier, Port Louis",
        is_sanctioned=True,
    )


@pytest.fixture
def sample_po() -> PurchaseOrder:
    """Approved Purchase Order matching $5,000 baseline."""
    return PurchaseOrder(
        po_id="PO-9001",
        vendor_id="VEND-001",
        amount=5000.00,
        currency="USD",
        created_by="EMP-1001",
        approved_by="EMP-2001",
        cost_center="CC-MARKETING",
        category="SUPPLIES",
    )


@pytest.fixture
def sample_gr() -> GoodsReceipt:
    """Valid Goods Receipt confirming delivery."""
    return GoodsReceipt(
        gr_id="GR-9001",
        po_id="PO-9001",
        vendor_id="VEND-001",
        received_by="EMP-3001",
        delivered_quantity=10.0,
        unit_cost=500.0,
        total_received_amount=5000.00,
    )


@pytest.fixture
def clean_invoice(clean_vendor, sample_po) -> Invoice:
    """Clean legitimate invoice matching PO."""
    return Invoice(
        invoice_id="INV-1001",
        vendor_id=clean_vendor.vendor_id,
        po_reference=sample_po.po_id,
        net_amount=5000.00,
        total_amount=5000.00,
        bank_iban=clean_vendor.bank_iban,
        extraction_confidence=0.99,
        line_items=[
            InvoiceLineItem(
                line_number=1,
                item_description="Office supplies bulk",
                quantity=10.0,
                unit_price=500.0,
                total_amount=5000.00,
            )
        ],
    )


@pytest.fixture
def graph_engine() -> KnowledgeGraphEngine:
    """Graph engine populated with vendor and employee nodes."""
    ge = KnowledgeGraphEngine()
    ge.add_vendor(
        vendor_id="VEND-COLLUSIVE",
        name="Nexus Tech Services",
        tax_id="44-5566778",
        bank_iban="US55NEXUS0001",
        address="742 Evergreen Terrace, Springfield, OR",
        directors=["Marcus Colluder"],
    )
    ge.add_employee(
        employee_id="EMP-APPROVER-99",
        name="Homer Simpson",
        role="Senior AP Approver",
        address="742 Evergreen Terrace, Springfield, OR",  # Identical address!
        phone="+1-555-7334",
    )
    ge.link_director_to_address("Marcus Colluder", "742 Evergreen Terrace, Springfield, OR")
    return ge


@pytest.fixture
def tri_tier_pipeline(graph_engine) -> TriTierDetectionPipeline:
    """Fully wired Tri-Tier detection pipeline."""
    feature_store = UnifiedFeatureStore(graph_engine=graph_engine)
    return TriTierDetectionPipeline(feature_store=feature_store)
