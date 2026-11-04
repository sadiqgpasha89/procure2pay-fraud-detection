"""Sub-millisecond Deterministic Rule Engine for Tier 1 RBS."""
import time
from typing import Any, Dict, List, Optional
import structlog
from src.core.models import (
    GoodsReceipt,
    Invoice,
    PurchaseOrder,
    RecommendedAction,
    Tier1Result,
    Vendor,
)
from src.detection.tier1_rules.registry import RuleRegistry

logger = structlog.get_logger(__name__)


class Tier1RuleEngine:
    """High-speed deterministic policy enforcement engine (<1ms execution target)."""

    def __init__(self, registry: Optional[RuleRegistry] = None):
        self.registry = registry or RuleRegistry()

    def evaluate(
        self,
        invoice: Invoice,
        vendor: Optional[Vendor] = None,
        po: Optional[PurchaseOrder] = None,
        gr: Optional[GoodsReceipt] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tier1Result:
        """Evaluate all registered rules against transaction artifacts."""
        start_time = time.perf_counter()
        violations: List[str] = []

        rules = self.registry.get_all_rules()
        for rule in rules:
            try:
                violation = rule.evaluate(
                    invoice=invoice,
                    vendor=vendor,
                    po=po,
                    gr=gr,
                    context=context,
                )
                if violation:
                    violations.append(f"[{rule.rule_name}] {violation}")
            except Exception as e:
                logger.error("tier1.rule_eval_error", rule=rule.rule_name, error=str(e))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        is_blocked = len(violations) > 0

        action = None
        if is_blocked:
            action = RecommendedAction.BLOCK_TRANSACTION
            logger.warn(
                "tier1.hard_violations_detected",
                invoice_id=invoice.invoice_id,
                vendor_id=invoice.vendor_id,
                violations=violations,
                latency_ms=elapsed_ms
            )

        return Tier1Result(
            is_blocked=is_blocked,
            violations=violations,
            latency_ms=round(elapsed_ms, 3),
            action=action,
        )
