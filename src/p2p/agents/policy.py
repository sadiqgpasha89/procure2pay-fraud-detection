"""Deterministic action policy. The LLM never selects the action (spec §5.4 guardrail 3 — action isolation):
the recommendation is a pure function of Tier-1 violations, Tier-2 score and *verified* findings."""
from __future__ import annotations

from p2p.agents.schemas import STRONG, Finding
from p2p.common.config import get_settings
from p2p.common.models import Action, Tier1Result, Tier2Result


def decide_action(t1: Tier1Result, t2: Tier2Result | None, findings: list[Finding], incomplete: bool = False) -> tuple[Action, str]:
    s = get_settings()
    kinds = {f.kind for f in findings if f.kind != "NO_ANOMALY"}
    high = {f.kind for f in findings if f.severity == "HIGH" and f.kind != "NO_ANOMALY"}
    score = t2.composite_score if t2 else 0.0

    if "SANCTIONS_OFAC_HIT" in t1.violations:
        return Action.FREEZE_VENDOR, "Sanctions watchlist hit (statutory)."
    if kinds & STRONG:
        return Action.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT, f"Strong fraud indicator(s) corroborated by evidence: {sorted(kinds & STRONG)}."
    if "PROMPT_INJECTION_DETECTED" in kinds:
        return Action.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT, "Document contained instructions aimed at automated reviewers — treat as active manipulation attempt."
    if score >= s.TAU_2 and (high or t1.is_blocked):
        return Action.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT, "High composite risk with corroborating evidence."
    if t1.is_blocked or high:
        return Action.HOLD_PAYMENT, "Control violation or high-severity finding; payment held for analyst review."
    if kinds & {"NO_CONTRACT", "SHARED_ADDRESS_OR_PHONE", "BANK_CHANGE_RECENT", "SOLE_SOURCE_AWARD", "PO_VARIANCE", "AMOUNT_OUTLIER", "OTHER_ANOMALY"} or score >= s.TAU_2:
        return Action.REQUEST_MANUAL_KYC, "Medium-severity signals; request manual vendor/document verification."
    if incomplete:  # a gap in the evidence can never be read as "clean"
        return Action.REQUEST_MANUAL_KYC, "Investigation incomplete (specialist or tool failure); cannot recommend release."
    return Action.AUTO_APPROVE, "Elevated statistical score was not corroborated by ERP, graph or document evidence (analyst confirms)."
