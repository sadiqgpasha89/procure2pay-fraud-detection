"""Case management and Human-in-the-Loop triage endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
import structlog
from src.api.dependencies import get_case_manager
from src.hitl.case_manager import CaseDispositionRequest, CaseManager, CaseRecord

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/cases", tags=["Case Management"])


@router.get("", response_model=List[CaseRecord])
async def list_cases(
    status: Optional[str] = None,
    case_manager: CaseManager = Depends(get_case_manager),
) -> List[CaseRecord]:
    """List triage cases, optionally filtered by status (PENDING_REVIEW, DISPOSED)."""
    return case_manager.list_cases(status=status)


@router.get("/{case_id}", response_model=CaseRecord)
async def get_case_detail(
    case_id: str,
    case_manager: CaseManager = Depends(get_case_manager),
) -> CaseRecord:
    """Retrieve full case details, graph findings, narrative, and evidence manifest."""
    record = case_manager.get_case(case_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return record


@router.post("/{case_id}/disposition", response_model=CaseRecord)
async def record_case_disposition(
    case_id: str,
    disposition: CaseDispositionRequest,
    case_manager: CaseManager = Depends(get_case_manager),
) -> CaseRecord:
    """Submit formal human investigator review and sign-off."""
    try:
        return case_manager.record_disposition(case_id, disposition)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    except Exception as e:
        logger.error("api.disposition_failed", case_id=case_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
