"""SOLID Protocols and Interfaces for Procure-to-Pay Fraud Detection & Prevention."""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from src.core.models import (
    Invoice,
    PurchaseOrder,
    GoodsReceipt,
    Vendor,
    Tier1Result,
    Tier2Result,
    CaseDossier,
    GraphFindings,
    EvidenceCitation,
)


@runtime_checkable
class RuleProtocol(Protocol):
    """Protocol for Tier 1 Deterministic Rules (Strategy Pattern)."""
    rule_name: str
    description: str

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Evaluate rule. Return violation string if violated, else None."""
        ...


@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Protocol for Online/Offline Feature Store."""

    def get_online_features(self, vendor_id: str, invoice_id: str) -> Dict[str, float]:
        """Fetch pre-aggregated behavioral and network features in <5ms."""
        ...

    def write_features(self, entity_key: str, features: Dict[str, float]) -> None:
        """Write features to store."""
        ...

    def get_historical_spend(
        self,
        vendor_id: str,
        cost_center: str,
        category: str,
        lookback_days: int
    ) -> float:
        """Fetch cumulative spend for rolling window smurfing detection."""
        ...


@runtime_checkable
class ScorerProtocol(Protocol):
    """Protocol for Tier 2 Machine Learning Scorers."""

    def score(
        self,
        features: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> Tier2Result:
        """Execute probabilistic scoring, return risk score, classification, and SHAP factors."""
        ...


@runtime_checkable
class GraphEngineProtocol(Protocol):
    """Protocol for Graph Database & Network Collusion Inspection."""

    def analyze_entity_conflicts(
        self,
        vendor_id: str,
        approver_id: Optional[str] = None
    ) -> GraphFindings:
        """Inspect graph for shared addresses, phone numbers, circular loops."""
        ...

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add node or relationship to graph."""
        ...


@runtime_checkable
class StreamBusProtocol(Protocol):
    """Protocol for Distributed Stream Bus (Kafka / GCP Pub/Sub)."""

    def publish_event(self, topic: str, key: str, payload: Dict[str, Any]) -> str:
        """Publish event partitioned by key, return message_id."""
        ...

    def subscribe(self, topic: str, group_id: str, handler: Any) -> None:
        """Subscribe to topic."""
        ...


@runtime_checkable
class AuditLoggerProtocol(Protocol):
    """Protocol for Immutable WORM Forensic Audit Ledger."""

    def log_entry(
        self,
        event_type: str,
        record_id: str,
        payload: Dict[str, Any],
        actor: str = "SYSTEM"
    ) -> EvidenceCitation:
        """Record forensic audit log entry with SHA-256 hash chaining."""
        ...

    def verify_integrity(self) -> bool:
        """Validate cryptographic continuity of the entire audit chain."""
        ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM Gateway / Specialist Agent Invocations."""

    def generate_investigation(
        self,
        prompt: str,
        context: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute structured LLM investigation returning validated schema."""
        ...
