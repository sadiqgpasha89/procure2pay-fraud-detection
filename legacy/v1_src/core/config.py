"""Application configuration using Pydantic Settings."""
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Procure-to-Pay Fraud Detection Platform."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Runtime & Service Info
    APP_NAME: str = "Procure-to-Pay Fraud Detection & Prevention"
    APP_VERSION: str = "1.0.0"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Risk Scoring Thresholds (Tri-Tier Framework)
    # Tau 1: Boundary between Low Risk (Auto-approve) and Medium Risk (Investigation)
    TAU_1_LOW_RISK_THRESHOLD: float = 0.35
    # Tau 2: Boundary between Medium Risk and High Risk (Strict holds, High-Sev Triage)
    TAU_2_HIGH_RISK_THRESHOLD: float = 0.70

    # Policy Hard Limits
    BANK_MODIFICATION_FREEZE_HOURS: int = 72
    OCR_CONFIDENCE_THRESHOLD: float = 0.85
    PO_VARIANCE_TOLERANCE_PCT: float = 0.10  # 10% tolerance ceiling
    DUAL_APPROVAL_AMOUNT_THRESHOLD: float = 50000.00

    # Circuit Breakers & Resilience
    CIRCUIT_BREAKER_FAIL_MAX: int = 5
    CIRCUIT_BREAKER_RESET_TIMEOUT_SEC: float = 30.0
    FALLBACK_RISK_PENALTY: float = 0.15

    # Pluggable Architecture Adapters (Open Stack vs GCP Stack)
    STREAM_BUS_TYPE: Literal["memory", "kafka", "gcp_pubsub"] = "memory"
    FEATURE_STORE_TYPE: Literal["memory", "redis", "bigquery"] = "memory"
    WORM_LEDGER_TYPE: Literal["file", "gcs"] = "file"
    LLM_PROVIDER: Literal["mock", "vertex_ai", "litellm"] = "mock"
    GRAPH_ENGINE_TYPE: Literal["networkx", "neo4j"] = "networkx"

    # GCP Stack Configuration
    GCP_PROJECT_ID: str = "p2p-fraud-detection-prod"
    GCP_REGION: str = "us-central1"
    GCP_PUBSUB_TOPIC: str = "projects/p2p-fraud-detection-prod/topics/p2p-events"
    GCP_BIGQUERY_DATASET: str = "p2p_fraud_features"
    GCP_GCS_AUDIT_BUCKET: str = "p2p-forensic-audit-worm-bucket"
    GCP_VERTEX_MODEL_NAME: str = "gemini-1.5-flash"

    # Open Stack Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "secret_password"

    # WORM Ledger Local Path
    LOCAL_WORM_LEDGER_PATH: str = "data/worm_audit_ledger.jsonl"


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for cached application settings."""
    return Settings()
