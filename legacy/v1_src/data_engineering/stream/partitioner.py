"""Stream partitioning and deterministic idempotency generator.

Optimized for high-throughput (5,000+ ops/sec) with distributed partitioning
by (Vendor_Hash_ID, Entity_Code) and UUIDv5 idempotency key generation.
"""
import hashlib
import uuid
from typing import Tuple


def generate_idempotency_key(entity_code: str, vendor_id: str, invoice_id: str, amount: float) -> str:
    """Generate deterministic UUIDv5 idempotency key from immutable transaction attributes."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # Standard DNS namespace
    raw_payload = f"{entity_code.strip()}:{vendor_id.strip()}:{invoice_id.strip()}:{amount:.2f}"
    return str(uuid.uuid5(namespace, raw_payload))


class StreamPartitioner:
    """Partitioner determining message queue target partitions."""

    def __init__(self, num_partitions: int = 32):
        self.num_partitions = max(1, num_partitions)

    def compute_partition_key(self, vendor_id: str, entity_code: str) -> str:
        """Create composite partition key."""
        return f"{entity_code.strip().upper()}#{vendor_id.strip().upper()}"

    def get_partition_id(self, vendor_id: str, entity_code: str) -> int:
        """Compute consistent partition index using SHA-256 hash."""
        composite = self.compute_partition_key(vendor_id, entity_code)
        hash_bytes = hashlib.sha256(composite.encode("utf-8")).digest()
        # Convert first 4 bytes to integer
        int_val = int.from_bytes(hash_bytes[:4], byteorder="big")
        return int_val % self.num_partitions
