"""Immutable WORM (Write Once, Read Many) Forensic Audit Ledger for SOX §404 compliance."""
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional
import structlog
from src.core.config import get_settings
from src.core.models import EvidenceCitation

logger = structlog.get_logger(__name__)


class WORMAuditBlock:
    """Individual tamper-evident block in the forensic audit ledger."""

    def __init__(
        self,
        index: int,
        timestamp: str,
        event_type: str,
        record_id: str,
        payload: Dict[str, Any],
        previous_hash: str,
        actor: str = "SYSTEM",
    ):
        self.index = index
        self.timestamp = timestamp
        self.event_type = event_type
        self.record_id = record_id
        self.payload = payload
        self.actor = actor
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Compute cryptographic SHA-256 hash of this block's contents linked to previous block."""
        canonical_str = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "event_type": self.event_type,
                "record_id": self.record_id,
                "payload": self.payload,
                "actor": self.actor,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "record_id": self.record_id,
            "payload": self.payload,
            "actor": self.actor,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class ImmutableWORMLedger:
    """Cryptographic WORM ledger storing all model inputs, rationales, citations, and decisions."""

    GENESIS_HASH = "0" * 64

    def __init__(self, persistence_path: Optional[str] = None):
        self.settings = get_settings()
        self.persistence_path = persistence_path or self.settings.LOCAL_WORM_LEDGER_PATH
        self.chain: List[WORMAuditBlock] = []
        self._init_genesis_block()

    def _init_genesis_block(self):
        """Create genesis block if chain is empty."""
        genesis = WORMAuditBlock(
            index=0,
            timestamp="2026-01-01T00:00:00Z",
            event_type="GENESIS",
            record_id="GENESIS-0",
            payload={"system": "P2P Fraud Prevention Platform", "compliance": "SOX_404"},
            previous_hash=self.GENESIS_HASH,
        )
        self.chain.append(genesis)

    def log_entry(
        self,
        event_type: str,
        record_id: str,
        payload: Dict[str, Any],
        actor: str = "SYSTEM"
    ) -> EvidenceCitation:
        """Append an immutable audit entry and return a cryptographically verifiable citation."""
        prev_block = self.chain[-1]
        now_ts = datetime.utcnow().isoformat() + "Z"

        new_block = WORMAuditBlock(
            index=len(self.chain),
            timestamp=now_ts,
            event_type=event_type,
            record_id=record_id,
            payload=payload,
            previous_hash=prev_block.hash,
            actor=actor,
        )
        self.chain.append(new_block)

        # Write append-only to local disk file if configured
        if self.persistence_path:
            try:
                os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
                with open(self.persistence_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(new_block.to_dict()) + "\n")
            except Exception as e:
                logger.error("worm_ledger.write_failed", error=str(e))

        logger.info(
            "worm_ledger.entry_logged",
            index=new_block.index,
            event_type=event_type,
            record_id=record_id,
            block_hash=new_block.hash[:16]
        )

        return EvidenceCitation(
            source=event_type,
            record_id=record_id,
            verification_hash=new_block.hash
        )

    def verify_integrity(self) -> bool:
        """Validate SHA-256 cryptographic continuity from genesis to head block."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]

            if curr.previous_hash != prev.hash:
                logger.error(
                    "worm_ledger.integrity_breach.previous_hash_mismatch",
                    index=curr.index,
                    expected=prev.hash,
                    found=curr.previous_hash
                )
                return False

            if curr.calculate_hash() != curr.hash:
                logger.error(
                    "worm_ledger.integrity_breach.content_tampered",
                    index=curr.index,
                    recorded_hash=curr.hash,
                    calculated_hash=curr.calculate_hash()
                )
                return False

        return True


@lru_cache()
def get_worm_ledger() -> ImmutableWORMLedger:
    """Singleton getter for WORM forensic audit ledger."""
    return ImmutableWORMLedger()
