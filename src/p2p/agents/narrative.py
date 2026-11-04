"""Deterministic narrative template + output guardrails (citation enforcement, grounding check)."""
from __future__ import annotations

import json
import re

from p2p.common.models import Evidence

CITE = re.compile(r"\[Evidence:\s*([^\]]+)\]", re.IGNORECASE)
ID_TOKEN = re.compile(r"\b(?:INV|VEND|EMP|PO|GR|CASE|BCHG|EML|CTR|DN|POL)-[0-9A-Z]+\b")
AMOUNT = re.compile(r"\$\s?[\d,]+(?:\.\d+)?")


def template_narrative(facts: dict, findings: list[dict]) -> str:
    parts = [f"Invoice {facts.get('invoice_id')} (${facts.get('amount', 0):,.2f}) from vendor {facts.get('vendor_id')} "
             f"scored {facts.get('score', 0):.2f} ({facts.get('classification')})."]
    if facts.get("rule_violations"):
        parts.append("Deterministic controls tripped: " + ", ".join(facts["rule_violations"]) + ".")
    for f in findings:
        if f["kind"] != "NO_ANOMALY":
            cite = f" [Evidence: {f['evidence'][0]['record_id']}]" if f.get("evidence") else ""
            parts.append(f["claim"].rstrip(".") + "." + cite)
    if len(parts) == 1:
        parts.append("Investigation found no corroborating anomaly in ERP, graph or document evidence.")
    return " ".join(parts)


def _amounts(text: str) -> set[float]:
    return {round(float(a.replace("$", "").replace(",", "").strip()), 2) for a in AMOUNT.findall(text) if a.replace("$", "").replace(",", "").strip().replace(".", "", 1).isdigit()}


def check_narrative(narrative: str, manifest: list[Evidence], facts: dict, findings: list[dict]) -> list[str]:
    """Return violations; empty list = narrative is cited and grounded.

    1. every [Evidence: x] must reference a verified manifest record;
    2. every ID and dollar amount must already appear in the facts or verified findings (no invented entities/figures)."""
    problems: list[str] = []
    ids = {e.record_id for e in manifest}
    for c in CITE.findall(narrative):
        if c.strip() not in ids:
            problems.append(f"unverified citation '{c.strip()}'")
    claims = " ".join(f["claim"] for f in findings)
    allowed_text = json.dumps(facts) + " " + claims + " " + " ".join(ids)
    allowed_amounts = _amounts(claims) | {round(float(v), 2) for v in facts.values() if isinstance(v, (int, float))}
    body = CITE.sub("", narrative)
    for tok in sorted(set(ID_TOKEN.findall(body))):
        if tok not in allowed_text:
            problems.append(f"ungrounded identifier '{tok}'")
    for amt in sorted(_amounts(body) - allowed_amounts):
        problems.append(f"ungrounded amount '${amt:,.2f}'")
    return problems
