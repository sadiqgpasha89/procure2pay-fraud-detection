"""Build the RAG index from the warehouse document table: `python -m p2p.rag.ingest`.

In production this is a Cloud Run job triggered by a GCS/Pub/Sub event when a contract lands in the
document bucket (Document AI → text → chunk → embed → upsert)."""
from __future__ import annotations

import argparse
import time

from p2p.common.config import get_settings
from p2p.data.repository import Warehouse
from p2p.rag.index import HybridIndex, make_embedder


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=s.EMBEDDING_PROVIDER, choices=["lexical", "ollama", "vertex"])
    a = ap.parse_args()
    t0 = time.time()
    docs = Warehouse(s.warehouse_path, readonly=True).documents()
    idx = HybridIndex.build(docs, make_embedder(a.provider), a.provider)
    idx.save(s.rag_dir)
    kinds: dict[str, int] = {}
    for d in docs:
        kinds[d.doc_type] = kinds.get(d.doc_type, 0) + 1
    print(f"indexed {len(docs)} documents {kinds} → {len(idx.chunks)} chunks "
          f"(provider={idx.embed_provider}) in {time.time() - t0:.1f}s → {s.rag_dir}")


if __name__ == "__main__":
    main()
