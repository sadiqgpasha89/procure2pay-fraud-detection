"""Tier 1 — deterministic policy engine. Pure functions over (entities, features); ~microseconds.

Every rule is a small class registered in `DEFAULT_RULES`; add a rule by appending to the list (or
loading rules authored from investigator-labelled cases — the 24h feedback loop in spec §9).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import structlog

from p2p.common.config import Settings, get_settings
from p2p.common.models import Tier1Result
from p2p.features.builder import TxContext

log = structlog.get_logger(__name__)


class Rule(Protocol):
    name: str
    description: str

    def check(self, tx: TxContext) -> str | None: ...


@dataclass
class ExactDuplicateInvoice:
    name: str = "EXACT_DUPLICATE_INVOICE"
    description: str = "Same vendor and invoice number already submitted."

    def check(self, tx: TxContext) -> str | None:
        n = int(tx.features["exact_duplicate_count"])
        return f"Invoice number {tx.invoice.invoice_number} was already submitted by {tx.invoice.vendor_id} ({n} earlier)." if n else None


@dataclass
class BankChangeFreeze:
    hours: int = 72
    name: str = "BANK_MODIFIED_WITHIN_72_HOURS"
    description: str = "Vendor bank details changed inside the mandatory freeze window."

    def check(self, tx: TxContext) -> str | None:
        if tx.features["bank_changed_within_72h"]:
            return (f"Settlement bank account changed {tx.features['bank_age_days'] * 24:.1f}h before this invoice "
                    f"(freeze {self.hours}h); out-of-band dual authorisation required.")
        return None


@dataclass
class SanctionsHit:
    name: str = "SANCTIONS_OFAC_HIT"
    description: str = "Vendor appears on a sanctions / PEP watchlist."

    def check(self, tx: TxContext) -> str | None:
        v = tx.vendor
        return f"Vendor {v.vendor_id} ({v.legal_name}) is on an active sanctions watchlist." if v.is_sanctioned else None


@dataclass
class PoCeiling:
    tolerance: float = 0.10
    name: str = "PO_CEILING_TOLERANCE_EXCEEDED"
    description: str = "Invoice exceeds PO amount by more than the hard tolerance."

    def check(self, tx: TxContext) -> str | None:
        v = tx.features["po_variance_ratio"]
        if tx.po and v > self.tolerance:
            return (f"Invoice ${tx.invoice.total_amount:,.2f} exceeds PO {tx.po.po_id} (${tx.po.amount:,.2f}) by {v:.1%} "
                    f"(ceiling {self.tolerance:.0%}).")
        return None


@dataclass
class SplitInvoice:
    limit: float = 50_000.0
    name: str = "THRESHOLD_PROXIMITY_SPLIT_DETECTED"
    description: str = "Invoice sits under the dual-approval limit but same-day companions push the total over it."

    def check(self, tx: TxContext) -> str | None:
        if tx.features["smurfing_pattern"]:
            return (f"Invoice ${tx.invoice.total_amount:,.2f} is below ${self.limit:,.0f}, but 24h spend for this vendor/cost-centre is "
                    f"${tx.features['rolling_24h_spend_ratio'] * self.limit:,.2f}.")
        return None


def default_rules(s: Settings | None = None) -> list[Rule]:
    s = s or get_settings()
    return [ExactDuplicateInvoice(), BankChangeFreeze(s.BANK_FREEZE_HOURS), SanctionsHit(),
            PoCeiling(s.PO_VARIANCE_TOLERANCE), SplitInvoice(s.DUAL_APPROVAL_LIMIT)]


class RuleEngine:
    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules if rules is not None else default_rules()

    def evaluate(self, tx: TxContext) -> Tier1Result:
        t0 = time.perf_counter()
        names, msgs = [], []
        for r in self.rules:
            try:
                if (m := r.check(tx)):
                    names.append(r.name)
                    msgs.append(m)
            except Exception as exc:  # a broken rule must never take the payment gate down
                log.error("rule_error", rule=r.name, error=str(exc))
        return Tier1Result(is_blocked=bool(names), violations=names, messages=msgs,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 3))
