"""Unified Dual-Tier Feature Store (Online Redis/Memory + Offline BigQuery/Historical)."""
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional
import structlog
from src.core.config import get_settings
from src.core.interfaces import FeatureStoreProtocol
from src.core.models import Invoice, PurchaseOrder, Vendor
from src.data_engineering.feature_store.real_time_features import compute_real_time_features
from src.data_engineering.feature_store.behavioral_features import (
    HistoricalSpendProfile,
    compute_behavioral_features,
)
from src.data_engineering.feature_store.graph_features import KnowledgeGraphEngine

logger = structlog.get_logger(__name__)


class UnifiedFeatureStore:
    """Enterprise Feature Store unifying low-latency online cache and analytical offline aggregations."""

    def __init__(self, graph_engine: Optional[KnowledgeGraphEngine] = None):
        self.settings = get_settings()
        self.online_cache: Dict[str, Dict[str, float]] = {}
        self.historical_profiles: Dict[str, HistoricalSpendProfile] = {}
        self.graph_engine = graph_engine or KnowledgeGraphEngine()
        self.spend_records: List[Dict[str, Any]] = []

    def set_historical_profile(self, vendor_id: str, profile: HistoricalSpendProfile) -> None:
        """Register historical baseline for vendor."""
        self.historical_profiles[vendor_id] = profile

    def record_transaction_spend(
        self,
        vendor_id: str,
        invoice_id: str,
        amount: float,
        cost_center: str = "CC-100",
        category: str = "GENERAL",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record settled spend for rolling window calculations."""
        ts = timestamp or datetime.utcnow()
        self.spend_records.append({
            "vendor_id": vendor_id,
            "invoice_id": invoice_id,
            "amount": amount,
            "cost_center": cost_center,
            "category": category,
            "timestamp": ts,
        })

    def get_historical_spend(
        self,
        vendor_id: str,
        cost_center: str,
        category: str,
        lookback_hours: int = 24,
        reference_time: Optional[datetime] = None
    ) -> float:
        """Fetch cumulative spend for rolling window smurfing detection."""
        ref_time = reference_time or datetime.utcnow()
        cutoff = ref_time.timestamp() - (lookback_hours * 3600)

        total = 0.0
        for rec in self.spend_records:
            if rec["vendor_id"] == vendor_id and rec.get("cost_center") == cost_center:
                if rec["timestamp"].timestamp() >= cutoff:
                    total += rec["amount"]
        return total

    def assemble_features(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Assemble complete feature vector across real-time, behavioral, and graph domains."""
        ref_time = reference_time or datetime.utcnow()

        # 1. Real-time streaming features
        rt_feats = compute_real_time_features(invoice, vendor, po, reference_time=ref_time)

        # 2. Behavioral features
        history = self.historical_profiles.get(invoice.vendor_id)
        r24 = self.get_historical_spend(invoice.vendor_id, po.cost_center if po else "CC-100", "GEN", 24, ref_time)
        r7 = self.get_historical_spend(invoice.vendor_id, po.cost_center if po else "CC-100", "GEN", 168, ref_time)
        r30 = self.get_historical_spend(invoice.vendor_id, po.cost_center if po else "CC-100", "GEN", 720, ref_time)
        beh_feats = compute_behavioral_features(invoice, history, r24, r7, r30)

        # 3. Graph features
        graph_feats = self.graph_engine.compute_graph_metrics(invoice.vendor_id)

        # Merge
        merged = {**rt_feats, **beh_feats, **graph_feats}

        # Cache in online store (<5ms access)
        entity_key = f"{invoice.vendor_id}#{invoice.invoice_id}"
        self.online_cache[entity_key] = merged

        return merged

    def get_online_features(self, vendor_id: str, invoice_id: str) -> Dict[str, float]:
        """Fetch pre-computed features from online store."""
        entity_key = f"{vendor_id}#{invoice_id}"
        return self.online_cache.get(entity_key, {})

    def write_features(self, entity_key: str, features: Dict[str, float]) -> None:
        """Write features directly to online store."""
        self.online_cache[entity_key] = features


@lru_cache()
def get_feature_store() -> UnifiedFeatureStore:
    """Singleton getter for shared feature store."""
    return UnifiedFeatureStore()
