"""Content-addressed evidence.

An `Evidence` item is `(source, record_id, sha256(canonical_json(record)))`. Tool servers mint
them when they *read* a record; the investigator re-derives the hash through the same server
before a citation is allowed into the dossier. A model can therefore quote evidence only if the
underlying record really exists and is unchanged — it cannot invent a hash.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from p2p.common.models import Evidence


def _default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default)


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_evidence(source: str, record_id: str, payload: Any) -> Evidence:
    return Evidence(source=source, record_id=record_id, verification_hash=content_hash(payload))
