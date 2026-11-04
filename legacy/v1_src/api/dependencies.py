"""Dependency injection wiring for FastAPI endpoints."""
from functools import lru_cache
from src.core.config import Settings, get_settings
from src.data_engineering.audit.worm_ledger import ImmutableWORMLedger, get_worm_ledger
from src.data_engineering.feature_store.store import UnifiedFeatureStore, get_feature_store
from src.data_engineering.ingestion.entity_resolution import EntityResolver
from src.data_engineering.ingestion.ocr_extractor import OCRExtractor
from src.detection.pipeline import TriTierDetectionPipeline
from src.hitl.case_manager import CaseManager, get_case_manager
from src.mlops.drift_monitor import DriftMonitor


@lru_cache()
def get_pipeline() -> TriTierDetectionPipeline:
    return TriTierDetectionPipeline()


@lru_cache()
def get_ocr_extractor() -> OCRExtractor:
    return OCRExtractor()


@lru_cache()
def get_entity_resolver() -> EntityResolver:
    return EntityResolver()


@lru_cache()
def get_drift_monitor() -> DriftMonitor:
    dm = DriftMonitor()
    # Populate initial baseline
    dm.set_baseline("vendor_bank_account_age_days", [100.0, 150.0, 200.0, 300.0, 450.0, 500.0, 600.0, 120.0, 180.0, 220.0])
    dm.set_baseline("invoice_amount_vs_90d_avg_ratio", [1.0, 1.05, 0.95, 1.1, 0.9, 1.0, 1.02, 1.08, 0.92, 1.0])
    return dm
