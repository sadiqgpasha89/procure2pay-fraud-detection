"""Evaluation endpoint for real-time transaction scoring (<850ms latency target)."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog
from src.api.dependencies import get_pipeline, get_case_manager
from src.core.models import GoodsReceipt, Invoice, PurchaseOrder, Vendor
from src.detection.pipeline import DetectionOutcome, TriTierDetectionPipeline
from src.hitl.case_manager import CaseManager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])


class EvaluateTransactionRequest(BaseModel):
    invoice: Invoice
    vendor: Optional[Vendor] = None
    purchase_order: Optional[PurchaseOrder] = None
    goods_receipt: Optional[GoodsReceipt] = None
    approver_id: Optional[str] = None


@router.post("/evaluate", response_model=DetectionOutcome)
async def evaluate_transaction(
    request: EvaluateTransactionRequest,
    pipeline: TriTierDetectionPipeline = Depends(get_pipeline),
    case_manager: CaseManager = Depends(get_case_manager),
) -> DetectionOutcome:
    """Evaluate an expenditure transaction through the Tri-Tier Hybrid Detection Framework."""
    try:
        outcome = pipeline.evaluate_transaction(
            invoice=request.invoice,
            vendor=request.vendor,
            po=request.purchase_order,
            gr=request.goods_receipt,
            approver_id=request.approver_id,
        )

        # If a Case Dossier was synthesized, register it in the case manager for HITL review
        if outcome.case_dossier:
            case_manager.register_case(
                dossier=outcome.case_dossier,
                invoice_amount=request.invoice.total_amount
            )

        return outcome
    except Exception as e:
        logger.error("api.evaluation_failed", invoice_id=request.invoice.invoice_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Transaction evaluation failed: {str(e)}")
