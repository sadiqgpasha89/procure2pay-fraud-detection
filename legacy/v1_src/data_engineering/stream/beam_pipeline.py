"""Apache Beam / Google Cloud Dataflow streaming pipeline topology definition."""
from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)


class P2PStreamTransformation:
    """Core stream transformations for invoice-PO-GR streaming reconciliation and feature computation."""

    @staticmethod
    def extract_composite_key(event: Dict[str, Any]) -> str:
        """Extract partition key for stateful windowing."""
        vendor_id = event.get("vendor_id", "UNKNOWN")
        entity_code = event.get("entity_code", "ENT-01")
        return f"{entity_code}:{vendor_id}"

    @staticmethod
    def calculate_stream_variance(invoice: Dict[str, Any], po: Dict[str, Any]) -> float:
        """Calculate relative price variance between invoice and PO."""
        po_amt = float(po.get("amount", 0.0))
        inv_amt = float(invoice.get("total_amount", 0.0))
        if po_amt <= 0.0:
            return 1.0
        return abs(inv_amt - po_amt) / po_amt

    @staticmethod
    def detect_sliding_window_smurfing(
        window_invoices: List[Dict[str, Any]],
        dual_approval_threshold: float = 50000.0
    ) -> Dict[str, Any]:
        """Examine invoices within 24h sliding window for split-invoicing under approval limit."""
        total_window_spend = sum(float(inv.get("total_amount", 0.0)) for inv in window_invoices)
        count = len(window_invoices)
        has_split_pattern = False

        if count >= 2 and total_window_spend >= dual_approval_threshold:
            # Check if individual amounts were each just below the limit
            all_below = all(float(inv.get("total_amount", 0.0)) < dual_approval_threshold for inv in window_invoices)
            if all_below:
                has_split_pattern = True

        return {
            "window_spend": total_window_spend,
            "invoice_count": count,
            "split_detected": has_split_pattern,
        }


def build_beam_pipeline_definition(runner_type: str = "DirectRunner"):
    """Blueprint for creating Apache Beam Dataflow streaming pipeline."""
    # Dataflow topology manifest
    pipeline_spec = {
        "pipeline_name": "p2p-fraud-stream-processor",
        "runner": runner_type,
        "transforms": [
            {"step": "ReadFromPubSubOrKafka", "topic": "p2p-events"},
            {"step": "ParseAndValidateJson", "fn": "ParseJsonFn"},
            {"step": "AssignIdempotencyKey", "fn": "UUIDv5Generator"},
            {"step": "WindowIntoSlidingWindows", "duration_minutes": 1440, "slide_minutes": 60},
            {"step": "ComputeRealTimeFeatures", "fn": "FeatureExtractorFn"},
            {"step": "PublishToFeatureStoreAndEvaluator", "sink": "RedisAndPubSub"},
        ]
    }
    return pipeline_spec
