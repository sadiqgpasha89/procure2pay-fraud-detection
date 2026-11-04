"""Rule Registry managing pluggable deterministic rules (Strategy & Registry pattern)."""
from typing import Dict, List, Type
from src.core.interfaces import RuleProtocol
from src.detection.tier1_rules.rules import (
    ExactDuplicateInvoiceRule,
    BankChange72hFreezeRule,
    SanctionsOFACRule,
    POCeilingToleranceRule,
    SplitInvoicingHardRule,
)


class RuleRegistry:
    """Central registry of executable deterministic policy rules."""

    def __init__(self):
        self._rules: Dict[str, RuleProtocol] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        self.register(ExactDuplicateInvoiceRule())
        self.register(BankChange72hFreezeRule())
        self.register(SanctionsOFACRule())
        self.register(POCeilingToleranceRule())
        self.register(SplitInvoicingHardRule())

    def register(self, rule: RuleProtocol) -> None:
        """Register a new rule instance."""
        self._rules[rule.rule_name] = rule

    def unregister(self, rule_name: str) -> None:
        """Remove a rule by name."""
        self._rules.pop(rule_name, None)

    def get_rule(self, rule_name: str) -> RuleProtocol:
        return self._rules[rule_name]

    def get_all_rules(self) -> List[RuleProtocol]:
        return list(self._rules.values())
