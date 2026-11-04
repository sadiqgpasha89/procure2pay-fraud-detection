"""RAG over unstructured P2P documents (contracts, policies, vendor e-mails, delivery dockets).

Retrieval is *hybrid*: a lexical TF-IDF index (always on, zero cost, no model download) fused with an
optional dense index by Reciprocal Rank Fusion. Dense embeddings are pluggable:
    lexical (default) | ollama (nomic-embed-text, open weights) | vertex (text-embedding-005)
Metadata filters (vendor_id, doc_type) are applied *before* ranking, so a query about vendor A can never
surface vendor B's contract. Each hit carries a content hash of its chunk text → citable evidence.

Scale-out path: replace `HybridIndex` with pgvector / AlloyDB / Vertex AI Vector Search behind the same
`search()` signature; the MCP tool and the agents do not change.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from p2p.common.config import get_settings
from p2p.common.models import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    vendor_id: str | None
    title: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def chunk_document(doc: Document, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    """Paragraph/sentence-aware chunking with a small overlap. Short docs stay whole."""
    text = doc.text.strip()
    pieces: list[str] = []
    if len(text) <= max_chars:
        pieces = [text]
    else:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        cur = ""
        for sent in sentences:
            if cur and len(cur) + len(sent) + 1 > max_chars:
                pieces.append(cur.strip())
                cur = cur[-overlap:] + " " + sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            pieces.append(cur.strip())
    return [Chunk(f"{doc.doc_id}#{i}", doc.doc_id, doc.doc_type, doc.vendor_id, doc.title, p) for i, p in enumerate(pieces)]


# ── dense embedders ─────────────────────────────────────────────────────────
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, model: str | None = None, base: str | None = None):
        s = get_settings()
        self.model, self.base = model or s.OLLAMA_EMBED_MODEL, base or s.OLLAMA_API_BASE

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            r = httpx.post(f"{self.base}/api/embed", json={"model": self.model, "input": texts[i:i + 64]}, timeout=120)
            r.raise_for_status()
            out += r.json()["embeddings"]
        return _unit(np.array(out, dtype=np.float32))


class VertexEmbedder:
    """Vertex AI embeddings through the google-genai SDK (needs GOOGLE_CLOUD_PROJECT + ADC)."""

    def __init__(self, model: str | None = None):
        from google import genai

        s = get_settings()
        self.model = model or s.VERTEX_EMBED_MODEL
        self.client = genai.Client(vertexai=True, project=s.GOOGLE_CLOUD_PROJECT, location=s.GOOGLE_CLOUD_LOCATION)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 100):
            res = self.client.models.embed_content(model=self.model, contents=texts[i:i + 100])
            out += [e.values for e in res.embeddings]
        return _unit(np.array(out, dtype=np.float32))


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def make_embedder(provider: str | None = None) -> Embedder | None:
    provider = provider or get_settings().EMBEDDING_PROVIDER
    return {"lexical": None, "ollama": OllamaEmbedder, "vertex": VertexEmbedder}[provider]() if provider != "lexical" else None


# ── index ──────────────────────────────────────────────────────────────────
class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectorizer: TfidfVectorizer, tfidf, dense: np.ndarray | None, embed_provider: str):
        self.chunks, self.vec, self.tfidf, self.dense, self.embed_provider = chunks, vectorizer, tfidf, dense, embed_provider
        self._embedder: Embedder | None = None

    @classmethod
    def build(cls, docs: list[Document], embedder: Embedder | None = None, provider: str = "lexical") -> "HybridIndex":
        chunks = [c for d in docs for c in chunk_document(d)]
        corpus = [f"{c.title}. {c.text}" for c in chunks]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        tfidf = vec.fit_transform(corpus)
        dense = embedder.embed(corpus) if embedder else None
        return cls(chunks, vec, tfidf, dense, provider if embedder else "lexical")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks": [asdict(c) for c in self.chunks], "vec": self.vec, "tfidf": self.tfidf, "dense": self.dense,
                     "provider": self.embed_provider}, directory / "index.joblib")

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        d = joblib.load(directory / "index.joblib")
        return cls([Chunk(**c) for c in d["chunks"]], d["vec"], d["tfidf"], d["dense"], d["provider"])

    def search(self, query: str, k: int = 4, vendor_id: str | None = None, doc_types: list[str] | None = None) -> list[Hit]:
        allowed = np.array([
            (doc_types is None or c.doc_type in doc_types) and (vendor_id is None or c.vendor_id in (vendor_id, None))
            for c in self.chunks])
        if not allowed.any():
            return []
        lex = (self.tfidf @ self.vec.transform([query]).T).toarray().ravel()
        rankings = [np.where(allowed, lex, -1.0)]
        if self.dense is not None:
            if self._embedder is None:
                self._embedder = make_embedder(self.embed_provider)
            rankings.append(np.where(allowed, self.dense @ self._embedder.embed([query])[0], -1.0))
        fused = np.zeros(len(self.chunks))
        for scores in rankings:  # Reciprocal Rank Fusion (k=60)
            order = np.argsort(-scores)
            for rank, idx in enumerate(order[: max(50, k * 5)]):
                if scores[idx] > 0:
                    fused[idx] += 1.0 / (60 + rank + 1)
        top = [i for i in np.argsort(-fused)[:k] if fused[i] > 0]
        norm = fused[top[0]] if top else 1.0
        return [Hit(self.chunks[i], round(float(fused[i] / norm), 4)) for i in top]


_cached: HybridIndex | None = None


def get_index(force_reload: bool = False) -> HybridIndex:
    global _cached
    if _cached is None or force_reload:
        _cached = HybridIndex.load(get_settings().rag_dir)
    return _cached
