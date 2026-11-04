"""Domain models and schemas for Procure-to-Pay Fraud Detection & Prevention."""
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RiskClassification(str, Enum):
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"


class RecommendedAction(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    REQUEST_MANUAL_KYC = "REQUEST_MANUAL_KYC"
    HOLD_PAYMENT = "HOLD_PAYMENT"
    HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT = "HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT"
    FREEZE_VENDOR = "FREEZE_VENDOR"
    BLOCK_TRANSACTION = "BLOCK_TRANSACTION"


class Vendor(BaseModel):
    """Vendor Master Data representation."""
    model_config = ConfigDict(extra="ignore")

    vendor_id: str = Field(..., description="Unique enterprise vendor ID")
    legal_name: str = Field(..., description="Registered corporate entity name")
    tax_id: str = Field(..., description="Tax identification / EIN / VAT number")
    bank_iban: str = Field(..., description="Active settlement account IBAN or routing+acc")
    bank_routing: Optional[str] = None
    bank_account_updated_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() - timedelta(days=365),
        description="Timestamp of the most recent settlement bank modification"
    )
    registration_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date vendor was onboarded"
    )
    address: str = Field(..., description="Registered physical address")
    phone: Optional[str] = None
    email_domain: Optional[str] = None
    directors: List[str] = Field(default_factory=list, description="Corporate directors and officers")
    is_sanctioned: bool = Field(default=False, description="OFAC / PEP sanction flag")
    is_active: bool = Field(default=True)


class PurchaseOrder(BaseModel):
    """Purchase Order (PO) issued to vendor."""
    model_config = ConfigDict(extra="ignore")

    po_id: str = Field(..., description="Unique PO reference")
    vendor_id: str = Field(..., description="Target vendor ID")
    amount: float = Field(..., description="Total committed PO currency amount")
    currency: str = Field(default="USD")
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="Employee who authored the PO")
    approved_by: Optional[str] = Field(None, description="Employee who approved the PO")
    cost_center: str = Field(default="CC-100")
    category: str = Field(default="GENERAL_PROCUREMENT")
    line_items: List[Dict[str, Any]] = Field(default_factory=list)


class GoodsReceipt(BaseModel):
    """Goods/Service Receipt (GR) confirming delivery fulfillment."""
    model_config = ConfigDict(extra="ignore")

    gr_id: str = Field(..., description="Unique Goods Receipt ID")
    po_id: str = Field(..., description="Associated PO reference")
    vendor_id: str = Field(..., description="Vendor delivering goods")
    received_at: datetime = Field(default_factory=datetime.utcnow)
    received_by: str = Field(..., description="Warehouse/AP receiver employee ID")
    delivered_quantity: float
    unit_cost: float
    total_received_amount: float
    delivery_note_id: Optional[str] = None


class InvoiceLineItem(BaseModel):
    """Single line item on an invoice."""
    model_config = ConfigDict(extra="ignore")

    line_number: int
    item_description: str
    quantity: float
    unit_price: float
    total_amount: float


class Invoice(BaseModel):
    """Invoice submitted for payment settlement."""
    model_config = ConfigDict(extra="ignore")

    invoice_id: str = Field(..., description="Unique invoice number / ID")
    vendor_id: str = Field(..., description="Vendor identifier")
    po_reference: Optional[str] = Field(None, description="Linked Purchase Order ID")
    invoice_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    net_amount: float = Field(..., description="Total billing net amount")
    tax_amount: float = Field(default=0.0)
    total_amount: float = Field(..., description="Gross total billing amount")
    currency: str = Field(default="USD")
    bank_iban: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    line_items: List[InvoiceLineItem] = Field(default_factory=list)
    extraction_confidence: float = Field(
        default=1.0,
        description="Ce: OCR or ingestion layout analysis confidence score [0.0 - 1.0]"
    )
    raw_document_uri: Optional[str] = None
    entity_code: str = Field(default="ENT-01", description="Business unit / operating company")


class PaymentEvent(BaseModel):
    """Payment release candidate event triggered before disbursement."""
    model_config = ConfigDict(extra="ignore")

    idempotency_key: str = Field(..., description="Deterministic UUIDv5 idempotency key")
    payment_id: str = Field(..., description="Batch or transaction payment identifier")
    invoice_id: str
    vendor_id: str
    amount: float
    currency: str = "USD"
    bank_iban_routing: str
    payment_method: str = "ACH"  # ACH, WIRE, SEPA, SWIFT
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    entity_code: str = "ENT-01"


class EvidenceCitation(BaseModel):
    """Cryptographically verifiable evidence item."""
    source: str = Field(..., description="System origin (ERP_AUDIT_LOG, GRAPH_DB, OCR, etc.)")
    record_id: str = Field(..., description="Primary key or node/edge ID in origin system")
    verification_hash: str = Field(..., description="SHA-256 integrity hash of source record")


class ShapContribution(BaseModel):
    """Feature contribution attribution from TreeSHAP."""
    feature: str
    value: float
    shap_value: float


class RiskAssessment(BaseModel):
    """Structured risk score and contributing factors."""
    composite_risk_score: float = Field(..., ge=0.0, le=1.0)
    classification: RiskClassification
    rule_violations: List[str] = Field(default_factory=list)
    top_contributing_factors: List[ShapContribution] = Field(default_factory=list)


class GraphFindings(BaseModel):
    """Knowledge graph collusion and network anomaly findings."""
    entity_conflicts_detected: bool = False
    details: str = "No conflicting graph linkages detected."
    collusion_nodes: List[str] = Field(default_factory=list)


class TransactionIdentifiers(BaseModel):
    """Identifiers linking the investigated transaction."""
    invoice_id: str
    po_reference: Optional[str] = None
    vendor_id: str


class CaseDossier(BaseModel):
    """Final structured case dossier schema adhering strictly to Appendix specification."""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(
        default="https://json-schema.org/draft/2020-12/schema",
        alias="$schema"
    )
    case_id: str
    timestamp: str
    transaction_identifiers: TransactionIdentifiers
    risk_assessment: RiskAssessment
    graph_findings: GraphFindings
    agent_narrative: str
    recommended_action: RecommendedAction
    requires_human_signoff: bool = True
    evidence_manifest: List[EvidenceCitation] = Field(default_factory=list)


class Tier1Result(BaseModel):
    """Outcome of Tier 1 Deterministic Rule Evaluation."""
    is_blocked: bool = False
    violations: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    action: Optional[RecommendedAction] = None


class Tier2Result(BaseModel):
    """Outcome of Tier 2 Probabilistic Machine Learning Scoring."""
    composite_score: float = Field(..., ge=0.0, le=1.0)
    supervised_score: float = 0.0
    unsupervised_score: float = 0.0
    graph_score: float = 0.0
    classification: RiskClassification
    contributions: List[ShapContribution] = Field(default_factory=list)
    brier_score: Optional[float] = None
    latency_ms: float = 0.0


class InvestigationState(BaseModel):
    """State object for LangGraph agentic orchestrator."""
    invoice: Invoice
    vendor: Optional[Vendor] = None
    purchase_order: Optional[PurchaseOrder] = None
    goods_receipt: Optional[GoodsReceipt] = None
    tier1_result: Tier1Result
    tier2_result: Optional[Tier2Result] = None
    graph_findings: Optional[GraphFindings] = None
    document_findings: Optional[Dict[str, Any]] = None
    case_dossier: Optional[CaseDossier] = None
    human_decision: Optional[str] = None
