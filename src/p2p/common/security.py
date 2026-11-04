"""PII minimisation and untrusted-text hygiene, applied *at the tool boundary* so no agent — whatever
model sits behind it — ever receives raw tax IDs / IBANs or live instructions from retrieved documents."""
from __future__ import annotations

import re

_INJECTION = re.compile(
    r"(ignore (all )?(the )?(previous|prior|above) (instructions|prompts?)|disregard (all )?(previous|prior) |system (notice|prompt|message)"
    r"|you are now|as an ai (reviewer|assistant)|do not (tell|inform|alert)|mark (this|the) vendor as verified|recommend auto[_ ]approve"
    r"|override (the )?(policy|controls?))",
    re.IGNORECASE,
)


def mask_tax_id(v: str | None) -> str:
    return "***-**-" + v[-4:] if v and len(v) >= 4 else "TAX-***"


def mask_iban(v: str | None) -> str:
    return f"{v[:4]}****{v[-4:]}" if v and len(v) >= 8 else "IBAN-***"


def mask_phone(v: str | None) -> str:
    return "***-***-" + v[-4:] if v and len(v) >= 4 else "***"


def mask_text(text: str) -> str:
    text = re.sub(r"\b\d{2}-\d{7}\b", "**-*******", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", text)
    return re.sub(r"\bUS\d{2}BANK\d{8,}\b", lambda m: mask_iban(m.group(0)), text)


def neutralise_untrusted(text: str) -> tuple[str, bool]:
    """Strip instruction-like spans from retrieved/untrusted text. Returns (safe_text, was_suspicious).

    Heuristic defence-in-depth (the main defence is architectural: the model has read-only tools and cannot
    choose the final action). Suspicious content is also reported to the investigator as a finding."""
    hits = _INJECTION.findall(text)
    if not hits:
        return text, False
    safe = re.sub(r"\[?SYSTEM NOTICE[^\]]*\]?", "[REMOVED: instruction-like text in document]", text, flags=re.IGNORECASE)
    safe = _INJECTION.sub("[REMOVED]", safe)
    return safe, True
