"""Single source of runtime configuration (12-factor: everything is an env var).

Defaults describe the *open-stack* deployment (SQLite, NumPy vector store, Ollama or offline
reference model, JSONL ledger) so the whole system runs on a laptop for ~zero cost. Setting the
`*_PROVIDER` / `*_BACKEND` variables switches individual components to Google Cloud services.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── runtime ────────────────────────────────────────────────────────────────
    ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    DATA_DIR: Path = Path("data/runtime")  # sqlite warehouse, vector index, models, ledger
    SERVICE_TOKEN: str = ""  # shared bearer token for service-to-service calls ("" = auth off, dev only)

    # ── service topology (override per environment; on Cloud Run these are service URLs) ──
    SCORING_URL: str = "http://127.0.0.1:8101"
    CASE_URL: str = "http://127.0.0.1:8102"
    AUDIT_URL: str = "http://127.0.0.1:8103"
    GATEWAY_URL: str = "http://127.0.0.1:8100"
    MCP_ERP_URL: str = "http://127.0.0.1:8201/mcp"
    MCP_GRAPH_URL: str = "http://127.0.0.1:8202/mcp"
    MCP_KNOWLEDGE_URL: str = "http://127.0.0.1:8203/mcp"
    GRAPH_ANALYST_URL: str = "http://127.0.0.1:8301"
    DOC_AUDITOR_URL: str = "http://127.0.0.1:8302"
    INVESTIGATOR_URL: str = "http://127.0.0.1:8300"

    # ── decision policy (Tier 1/2) ─────────────────────────────────────────────
    TAU_1: float = 0.35  # below → auto-approve
    TAU_2: float = 0.70  # at/above → high risk
    BANK_FREEZE_HOURS: int = 72
    PO_VARIANCE_TOLERANCE: float = 0.10
    DUAL_APPROVAL_LIMIT: float = 50_000.0
    OCR_CONFIDENCE_MIN: float = 0.85
    FALLBACK_RISK_PENALTY: float = 0.15
    CIRCUIT_FAIL_MAX: int = 5
    CIRCUIT_RESET_SEC: float = 30.0

    # ── Tier 3 LLM ─────────────────────────────────────────────────────────────
    # offline : deterministic reference model, no network / no key (CI, demos, cold-start)
    # ollama  : open-weights model via LiteLLM (cheapest "real" LLM)
    # vertex  : Gemini on Vertex AI
    LLM_PROVIDER: Literal["offline", "ollama", "vertex"] = "offline"
    GEMINI_MODEL: str = "gemini-3.5-flash"
    OLLAMA_MODEL: str = "ollama_chat/gemma3:12b"
    OLLAMA_API_BASE: str = "http://127.0.0.1:11434"
    OLLAMA_THINK: bool = False  # reasoning/"thinking" mode burns the output-token budget and truncates JSON (measured 15s→2.5s per call with it off)
    # Forced first-hop tool calls issued by a callback instead of the model. Small open models often skip tools when thinking is off;
    # the first hop is deterministic anyway (same ERP/graph lookups for every invoice), so the LLM only has to interpret results.
    # auto = on for ollama, off for vertex (Gemini plans tool use reliably)
    TOOL_PREFETCH: Literal["auto", "on", "off"] = "auto"
    INVESTIGATION_TIMEOUT_SEC: float = 60.0

    # ── RAG ────────────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: Literal["lexical", "ollama", "vertex"] = "lexical"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    VERTEX_EMBED_MODEL: str = "text-embedding-005"
    RAG_TOP_K: int = 4

    # ── GCP (only read when a *_BACKEND is switched to gcp) ───────────────────────
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    AUDIT_SINK: Literal["file", "gcs"] = "file"
    AUDIT_GCS_BUCKET: str = ""
    AUDIT_BQ_TABLE: str = ""  # project.dataset.table, optional analytics sink

    # ── derived paths ──────────────────────────────────────────────────────────
    @property
    def warehouse_path(self) -> Path:
        return self.DATA_DIR / "warehouse.sqlite"

    @property
    def model_dir(self) -> Path:
        return self.DATA_DIR / "models"

    @property
    def rag_dir(self) -> Path:
        return self.DATA_DIR / "rag"

    @property
    def ledger_path(self) -> Path:
        return self.DATA_DIR / "audit" / "ledger.jsonl"

    @property
    def case_db_url(self) -> str:
        return f"sqlite:///{(self.DATA_DIR / 'cases.sqlite').resolve()}"

    @property
    def decision_db_path(self) -> Path:
        return self.DATA_DIR / "decisions.sqlite"

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.SERVICE_TOKEN}"} if self.SERVICE_TOKEN else {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
