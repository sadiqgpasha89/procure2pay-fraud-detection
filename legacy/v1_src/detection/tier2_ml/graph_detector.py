"""Graph collusion and circular loop detector for Tier 2 ML scoring."""
from typing import Any, Dict, Optional
import structlog
from src.core.models import GraphFindings
from src.data_engineering.feature_store.graph_features import KnowledgeGraphEngine

logger = structlog.get_logger(__name__)


class GraphCollusionDetector:
    """Network-level collusion and circular payment loop scorer."""

    def __init__(self, graph_engine: Optional[KnowledgeGraphEngine] = None):
        self.graph_engine = graph_engine or KnowledgeGraphEngine()

    def score_network_risk(
        self,
        vendor_id: str,
        approver_id: Optional[str] = None
    ) -> float:
        """Compute network risk score in [0.0, 1.0]."""
        findings = self.graph_engine.analyze_entity_conflicts(vendor_id, approver_id)
        metrics = self.graph_engine.compute_graph_metrics(vendor_id)

        risk = 0.0
        if findings.entity_conflicts_detected:
            risk += 0.50

        if metrics.get("shared_bank_account_cross_vendor_flag", 0.0) == 1.0:
            risk += 0.30

        if metrics.get("shared_contact_attribute_flag", 0.0) == 1.0:
            risk += 0.20

        centrality = metrics.get("degree_centrality_vendor", 0.0)
        if centrality > 0.25:
            risk += 0.15

        return min(1.0, risk)

    def get_findings(self, vendor_id: str, approver_id: Optional[str] = None) -> GraphFindings:
        """Retrieve structured graph inspection findings."""
        return self.graph_engine.analyze_entity_conflicts(vendor_id, approver_id)
