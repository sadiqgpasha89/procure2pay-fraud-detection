"""Tamper-evident forensic ledger (SOX §404): SHA-256 hash chain, single writer, pluggable WORM sinks.

  file : append-only JSONL (dev)
  gcs  : one create-only object per block in a bucket with a *locked retention policy* → true WORM
  bq   : optional analytics mirror (partitioned table) — query-only, never the source of truth
The chain is verified on startup and on demand; any edit/deletion/re-ordering breaks every later hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

GENESIS = "0" * 64


def block_hash(b: dict[str, Any]) -> str:
    body = {k: b[k] for k in ("index", "timestamp", "event_type", "record_id", "actor", "payload", "previous_hash")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class Sink(Protocol):
    def write(self, block: dict[str, Any]) -> None: ...
    def read_all(self) -> list[dict[str, Any]]: ...


class FileSink:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, block: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(block, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GcsSink:
    """Cloud Storage WORM sink. Create the bucket with a retention policy and lock it (`gsutil retention lock`)."""

    def __init__(self, bucket: str, prefix: str = "ledger/"):
        from google.cloud import storage

        self.bucket, self.prefix = storage.Client().bucket(bucket), prefix

    def write(self, block: dict[str, Any]) -> None:
        blob = self.bucket.blob(f"{self.prefix}{block['index']:012d}.json")
        blob.upload_from_string(json.dumps(block, sort_keys=True, default=str), content_type="application/json", if_generation_match=0)

    def read_all(self) -> list[dict[str, Any]]:
        blobs = sorted(self.bucket.client.list_blobs(self.bucket, prefix=self.prefix), key=lambda b: b.name)
        return [json.loads(b.download_as_text()) for b in blobs]


class BigQuerySink:
    def __init__(self, table: str):
        from google.cloud import bigquery

        self.client, self.table = bigquery.Client(), table

    def write(self, block: dict[str, Any]) -> None:
        row = {**block, "payload": json.dumps(block["payload"], default=str)}
        errs = self.client.insert_rows_json(self.table, [row])
        if errs:
            raise RuntimeError(f"BigQuery insert failed: {errs}")

    def read_all(self) -> list[dict[str, Any]]:
        return []


class Ledger:
    def __init__(self, primary: Sink, mirrors: list[Sink] | None = None):
        self.primary, self.mirrors, self._lock = primary, mirrors or [], threading.Lock()
        self.chain = primary.read_all()
        ok, why = self.verify()
        if not ok:
            raise RuntimeError(f"ledger integrity failure on startup: {why}")

    def append(self, event_type: str, record_id: str, payload: dict[str, Any], actor: str = "SYSTEM") -> dict[str, Any]:
        with self._lock:
            prev = self.chain[-1]["hash"] if self.chain else GENESIS
            b = {"index": len(self.chain), "timestamp": datetime.now(timezone.utc).isoformat(), "event_type": event_type, "record_id": record_id,
                 "actor": actor, "payload": payload, "previous_hash": prev}
            b["hash"] = block_hash(b)
            self.primary.write(b)  # durable first; a mirror failure must not lose the record
            self.chain.append(b)
        for m in self.mirrors:
            try:
                m.write(b)
            except Exception:  # noqa: BLE001 - mirrors are best-effort analytics copies
                pass
        return b

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, b in enumerate(self.chain):
            if b["index"] != i:
                return False, f"index gap at {i}"
            if b["previous_hash"] != prev:
                return False, f"broken link at block {i}"
            if block_hash(b) != b["hash"]:
                return False, f"tampered content at block {i}"
            prev = b["hash"]
        return True, "ok"
