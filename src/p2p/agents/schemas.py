"""Typed contracts between agents. Specialists must answer with a `SpecialistReport`; the closed `FindingKind`
vocabulary is what lets deterministic code (not the LLM) decide the recommended action."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from p2p.common.models import Evidence

FindingKind = Literal[
    "STAFF_VENDOR_LINK", "SHARED_BANK_ACCOUNT", "SHARED_ADDRESS_OR_PHONE", "BANK_CHANGE_EMAIL_LOOKALIKE", "BANK_CHANGE_RECENT",
    "NO_GOODS_RECEIPT", "SHORT_DELIVERY", "PO_VARIANCE", "SOD_VIOLATION", "DUPLICATE_INVOICE", "SPLIT_COMPANIONS", "AMOUNT_OUTLIER",
    "CONTRACT_PRICE_EXCEEDED", "NO_CONTRACT", "SOLE_SOURCE_AWARD", "PROMPT_INJECTION_DETECTED", "OTHER_ANOMALY", "NO_ANOMALY",
]
STRONG: set[str] = {"STAFF_VENDOR_LINK", "SHARED_BANK_ACCOUNT", "BANK_CHANGE_EMAIL_LOOKALIKE"}
Severity = Literal["HIGH", "MEDIUM", "INFO"]


class Finding(BaseModel):
    kind: FindingKind
    severity: Severity = "MEDIUM"
    claim: str = Field(description="One factual sentence. Every number/ID must come from tool output.")
    evidence: list[Evidence] = Field(default_factory=list, description="Copy source/record_id/verification_hash verbatim from tool results.")


class SpecialistReport(BaseModel):
    agent: str
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def repair_truncated_json(text: str) -> str | None:
    """Close brackets/quotes a model left open at the end of its output. Returns None if the text is not repairable by appending.
    Safe because the result is still schema-validated and every citation is hash-verified downstream."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            esc, in_str = (not esc and ch == "\\"), not (ch == '"' and not esc)
            if ch != "\\":
                esc = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
    if not stack and not in_str:
        return None
    return text + ('"' if in_str else "") + "".join(reversed(stack))


def parse_report_ex(text: str, agent: str) -> tuple[SpecialistReport | None, list[str]]:
    """Tolerant extraction + per-finding validation. A model that invents one off-vocabulary `kind` must not lose its other findings:
    such a finding is kept as OTHER_ANOMALY (never above MEDIUM weight in policy unless the model marked it HIGH) and reported as a problem."""
    problems: list[str] = []
    cleaned = _FENCE.sub("", (text or "").strip())
    start = cleaned.find("{")
    if start < 0:
        return None, ["no JSON object in output"]
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        fixed = repair_truncated_json(cleaned[start:])
        try:
            obj, _ = json.JSONDecoder().raw_decode(fixed) if fixed else (None, 0)
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict):
            return None, [f"invalid JSON ({exc.msg} at char {exc.pos}; output {len(cleaned)} chars — truncated?)"]
        problems.append("repaired truncated JSON (model left brackets open)")
    if not isinstance(obj, dict):
        return None, ["JSON root is not an object"]
    findings: list[Finding] = []
    for raw in obj.get("findings") or []:
        try:
            findings.append(Finding.model_validate(raw))
            continue
        except ValidationError:
            pass
        if isinstance(raw, dict) and raw.get("claim"):
            ev = []
            for e in raw.get("evidence") or []:
                try:
                    ev.append(Evidence.model_validate(e))
                except ValidationError:
                    problems.append("dropped malformed evidence item")
            sev = raw.get("severity") if raw.get("severity") in ("HIGH", "MEDIUM", "INFO") else "MEDIUM"
            try:
                findings.append(Finding(kind="OTHER_ANOMALY", severity=sev, claim=str(raw["claim"]), evidence=ev))
                problems.append(f"coerced unknown finding kind {raw.get('kind')!r} to OTHER_ANOMALY")
            except ValidationError:
                problems.append("dropped unparseable finding")
        else:
            problems.append("dropped finding without a claim")
    return SpecialistReport(agent=str(obj.get("agent") or agent), summary=str(obj.get("summary") or ""), findings=findings), problems


def parse_report(text: str, agent: str) -> SpecialistReport | None:
    return parse_report_ex(text, agent)[0]


REPORT_SCHEMA_HINT = json.dumps(SpecialistReport.model_json_schema(), separators=(",", ":"))
