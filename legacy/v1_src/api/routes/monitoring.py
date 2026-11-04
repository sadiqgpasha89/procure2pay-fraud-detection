"""Monitoring, drift evaluation, and operational health endpoints."""
from typing import Any, Dict
from fastapi import APIRouter, Depends
import structlog
from src.api.dependencies import get_case_manager, get_drift_monitor
from src.core.telemetry import global_metrics
from src.hitl.case_manager import CaseManager
from src.mlops.drift_monitor import DatasetDriftReport, DriftMonitor

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Monitoring & System Health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Liveness probe for container orchestrators."""
    return {"status": "HEALTHY", "service": "p2p-fraud-detection"}


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness probe verifying engine availability."""
    return {"status": "READY", "version": "1.0.0"}


@router.get("/api/v1/monitoring/metrics")
async def get_system_metrics(case_manager: CaseManager = Depends(get_case_manager)) -> Dict[str, Any]:
    """Retrieve comprehensive system, business, and LLMOps metrics snapshot."""
    telemetry_snap = global_metrics.get_all_metrics()
    llmops_metrics = case_manager.llm_metrics.get_metrics().model_dump()

    return {
        "business_metrics": {
            "gross_prevented_loss_usd": case_manager.prevented_loss_total,
            "total_triage_cases": len(case_manager.cases),
        },
        "system_telemetry": telemetry_snap,
        "llmops_metrics": llmops_metrics,
    }


@router.get("/api/v1/monitoring/drift", response_model=DatasetDriftReport)
async def evaluate_drift(drift_monitor: DriftMonitor = Depends(get_drift_monitor)) -> DatasetDriftReport:
    """Run Population Stability Index (PSI) and statistical drift analysis."""
    return drift_monitor.evaluate_all()
