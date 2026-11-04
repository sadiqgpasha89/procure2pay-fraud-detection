"""Domain entities and cross-service contracts.

Everything that crosses a process boundary (REST, MCP, A2A) is one of these pydantic models,
so every service validates the same schema. The Case Dossier follows the Appendix of the spec.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── enums ────────────────────────────────────────────────────────────────────
class RiskClass(str, Enum):
    LOW = "LOW_RISK"
    MEDIUM = "MEDIUM_RISK"
    HIGH = "HIGH_RISK"


class Action(str, Enum):
    """The only actions the platform can recommend. Agents never emit free-form actions."""

    AUTO_APPROVE = "AUTO_APPROVE"
    REQUEST_MANUAL_KYC = "REQUEST_MANUAL_KYC"
    HOLD_PAYMENT = "HOLD_PAYMENT"
    HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT = "HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT"
    FREEZE_VENDOR = "FREEZE_VENDOR"
    BLOCK_TRANSACTION = "BLOCK_TRANSACTION"


class Route(str, Enum):
    """Where the scoring service sends a transaction next."""

    APPROVE = "APPROVE"  # Tier 2 low risk
    INVESTIGATE = "INVESTIGATE"  # Tier 2 medium/high → Tier 3
    BLOCK = "BLOCK"  # Tier 1 hard violation


class CaseStatus(str, Enum):
    INVESTIGATING = "INVESTIGATING"
    PENDING_REVIEW = "PENDING_REVIEW"
    DISPOSED = "DISPOSED"


# ── master / transactional data ──────────────────────────────────────────────
class _Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Vendor(_Entity):
    vendor_id: str
    legal_name: str
    tax_id: str
    jurisdiction: str = "US-TX"
    bank_iban: str
    bank_account_updated_at: datetime
    registration_date: datetime
    address: str
    phone: str | None = None
    email_domain: str | None = None
    directors: list[str] = Field(default_factory=list)
    is_sanctioned: bool = False
    is_active: bool = True


class Employee(_Entity):
    employee_id: str
    name: str
    role: str  # BUYER | AP_CLERK | APPROVER | WAREHOUSE
    address: str
    phone: str | None = None
    department: str = "PROCUREMENT"


class PurchaseOrder(_Entity):
    po_id: str
    vendor_id: str
    amount: float
    currency: str = "USD"
    issued_at: datetime
    created_by: str
    approved_by: str | None = None
    cost_center: str = "CC-100"
    category: str = "GENERAL"
    item_code: str = "ITEM-GEN"
    quantity: float = 1.0
    unit_price: float = 0.0


class GoodsReceipt(_Entity):
    gr_id: str
    po_id: str
    vendor_id: str
    received_at: datetime
    received_by: str
    received_amount: float
    delivery_note_id: str | None = None


class InvoiceLine(_Entity):
    item_code: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(_Entity):
    invoice_id: str
    invoice_number: str  # vendor-facing number printed on the document
    vendor_id: str
    po_id: str | None = None
    invoice_date: datetime
    net_amount: float
    tax_amount: float = 0.0
    total_amount: float
    currency: str = "USD"
    bank_iban: str | None = None
    vendor_tax_id: str | None = None
    lines: list[InvoiceLine] = Field(default_factory=list)
    extraction_confidence: float = 1.0  # Ce from OCR / Document AI
    entity_code: str = "ENT-01"
    cost_center: str = "CC-100"
    category: str = "GENERAL"
    submitted_by: str | None = None  # AP clerk who keyed/approved the invoice


class BankChange(_Entity):
    change_id: str
    vendor_id: str
    changed_at: datetime
    old_iban: str
    new_iban: str
    requested_via: str = "VENDOR_PORTAL"  # VENDOR_PORTAL | EMAIL | PHONE
    requested_from: str | None = None


class Document(_Entity):
    doc_id: str
    doc_type: str  # CONTRACT | POLICY | EMAIL | DOCKET
    vendor_id: str | None = None
    title: str
    text: str
    created_at: datetime | None = None


# ── evidence & dossier ───────────────────────────────────────────────────────
class Evidence(BaseModel):
    """A verifiable pointer to a source record: sha256 of the canonical record at read time."""

    source: str  # ERP | GRAPH_DB | KNOWLEDGE_BASE
    record_id: str
    verification_hash: str


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class Tier1Result(BaseModel):
    is_blocked: bool = False
    violations: list[str] = Field(default_factory=list)  # rule names
    messages: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class Tier2Result(BaseModel):
    composite_score: float = Field(ge=0.0, le=1.0)
    supervised_score: float = 0.0
    anomaly_score: float = 0.0
    graph_score: float = 0.0
    classification: RiskClass
    contributions: list[ShapContribution] = Field(default_factory=list)
    fallback_penalty: float = 0.0
    model_version: str = ""
    latency_ms: float = 0.0


class ScoringResult(BaseModel):
    """Output of the scoring microservice (Tier 1 + Tier 2)."""

    invoice_id: str
    vendor_id: str
    amount: float = 0.0
    po_reference: str | None = None
    route: Route
    tier1: Tier1Result
    tier2: Tier2Result | None = None
    features: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0


class GraphFindings(BaseModel):
    entity_conflicts_detected: bool = False
    details: str = "No conflicting graph linkages detected."
    collusion_nodes: list[str] = Field(default_factory=list)


class TransactionIdentifiers(BaseModel):
    invoice_id: str
    po_reference: str | None = None
    vendor_id: str


class RiskAssessment(BaseModel):
    composite_risk_score: float = Field(ge=0.0, le=1.0)
    classification: RiskClass
    rule_violations: list[str] = Field(default_factory=list)
    top_contributing_factors: list[ShapContribution] = Field(default_factory=list)


class CaseDossier(BaseModel):
    """Automated Triage Output Schema (Appendix of the specification)."""

    model_config = ConfigDict(populate_by_name=True)

    schema_uri: str = Field(default="https://json-schema.org/draft/2020-12/schema", alias="$schema")
    case_id: str
    timestamp: str
    transaction_identifiers: TransactionIdentifiers
    risk_assessment: RiskAssessment
    graph_findings: GraphFindings
    document_findings: list[str] = Field(default_factory=list)
    agent_narrative: str
    recommended_action: Action
    requires_human_signoff: bool = True
    evidence_manifest: list[Evidence] = Field(default_factory=list)
    guardrail_notes: list[str] = Field(default_factory=list)
    llm_provider: str = ""
    synthesis_latency_sec: float = 0.0


class InvestigationRequest(BaseModel):
    """Message the gateway sends to the investigator agent over A2A."""

    case_id: str
    invoice_id: str
    vendor_id: str
    amount: float
    po_reference: str | None = None
    scoring: ScoringResult


class Decision(BaseModel):
    """What the gateway returns to the caller (ERP / payment run)."""

    invoice_id: str
    vendor_id: str
    idempotency_key: str
    final_action: Action
    route: Route
    scoring: ScoringResult
    case_id: str | None = None
    dossier: CaseDossier | None = None  # populated when wait_for_investigation=true and ready
    requires_human_signoff: bool = False
    audit_hash: str = ""
    total_latency_ms: float = 0.0


class Disposition(BaseModel):
    decision: str  # CONFIRMED_FRAUD | FALSE_POSITIVE | OVERRIDDEN
    analyst_id: str
    rationale: str
    accepted_recommendation: bool = True
