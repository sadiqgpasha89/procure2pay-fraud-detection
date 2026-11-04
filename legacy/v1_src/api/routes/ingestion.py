"""Ingestion endpoints for raw invoices, OCR documents, ERP webhooks, and vendor master data."""
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog
from src.api.dependencies import get_entity_resolver, get_ocr_extractor
from src.core.models import Invoice, Vendor
from src.data_engineering.ingestion.entity_resolution import EntityMatchResult, EntityResolver
from src.data_engineering.ingestion.erp_adapters import ERPAdapterFactory
from src.data_engineering.ingestion.ocr_extractor import OCRExtractor, OCRProcessingResult

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


class OCRDocumentRequest(BaseModel):
    raw_text: str
    document_uri: Optional[str] = None
    base_confidence: float = 0.95


class ERPIngestionRequest(BaseModel):
    erp_system: str  # e.g. "SAP", "NetSuite"
    payload: Dict[str, Any]


class VendorOnboardingResponse(BaseModel):
    vendor_id: str
    is_registered: bool
    potential_collisions: list[EntityMatchResult]


@router.post("/ocr", response_model=OCRProcessingResult)
async def ingest_ocr_document(
    request: OCRDocumentRequest,
    ocr_extractor: OCRExtractor = Depends(get_ocr_extractor),
) -> OCRProcessingResult:
    """Process unstructured document text via OCR and route to async queue if Ce < 0.85."""
    try:
        return ocr_extractor.process_raw_document(
            raw_text=request.raw_text,
            document_uri=request.document_uri,
            base_confidence=request.base_confidence,
        )
    except Exception as e:
        logger.error("api.ocr_ingestion_error", error=str(e))
        raise HTTPException(status_code=400, detail=f"OCR ingestion failed: {str(e)}")


@router.post("/erp", response_model=Invoice)
async def ingest_erp_payload(request: ERPIngestionRequest) -> Invoice:
    """Normalize SAP S/4HANA or Oracle NetSuite payload into canonical Invoice schema."""
    try:
        return ERPAdapterFactory.parse(request.erp_system, request.payload)
    except Exception as e:
        logger.error("api.erp_ingestion_error", error=str(e))
        raise HTTPException(status_code=400, detail=f"ERP parsing failed: {str(e)}")


@router.post("/vendor", response_model=VendorOnboardingResponse)
async def onboard_vendor(
    vendor: Vendor,
    resolver: EntityResolver = Depends(get_entity_resolver),
) -> VendorOnboardingResponse:
    """Onboard a vendor with multi-attribute fuzzy matching and imposter collision detection."""
    try:
        matches = resolver.find_matches(vendor, threshold=0.70)
        resolver.register_vendor(vendor)
        return VendorOnboardingResponse(
            vendor_id=vendor.vendor_id,
            is_registered=True,
            potential_collisions=matches,
        )
    except Exception as e:
        logger.error("api.vendor_onboarding_error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Vendor onboarding failed: {str(e)}")
