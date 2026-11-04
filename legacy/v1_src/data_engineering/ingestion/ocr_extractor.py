"""Document OCR and layout analysis extraction pipeline with confidence-based validation routing."""
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import structlog
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.models import Invoice, InvoiceLineItem

logger = structlog.get_logger(__name__)


class ExtractedField(BaseModel):
    """Field-level extraction output with confidence score."""
    value: Any
    confidence: float = Field(..., ge=0.0, le=1.0)


class OCRProcessingResult(BaseModel):
    """Result of invoice document OCR parsing."""
    invoice: Invoice
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    routed_to_validation_queue: bool = False
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""
    errors: List[str] = Field(default_factory=list)


class OCRExtractor:
    """Enterprise OCR & Layout Analysis Pipeline.
    
    Mitigates Edge Case 2: High OCR Error Rates by calculating field-level and document-level
    confidence scores Ce, routing documents with Ce < 0.85 to an async human validation queue.
    """

    def __init__(self, confidence_threshold: Optional[float] = None):
        self.settings = get_settings()
        self.confidence_threshold = confidence_threshold or self.settings.OCR_CONFIDENCE_THRESHOLD
        self.async_validation_queue: List[OCRProcessingResult] = []

    def process_raw_document(
        self,
        raw_text: str,
        document_uri: Optional[str] = None,
        base_confidence: float = 0.95
    ) -> OCRProcessingResult:
        """Parse raw OCR text or document layout into canonical Invoice schema."""
        field_confidences: Dict[str, float] = {}
        errors: List[str] = []

        # 1. Extract Invoice ID
        invoice_id, conf_inv = self._extract_invoice_id(raw_text, base_confidence)
        field_confidences["invoice_id"] = conf_inv

        # 2. Extract Vendor ID / Tax ID
        vendor_id, tax_id, conf_vendor = self._extract_vendor_identifiers(raw_text, base_confidence)
        field_confidences["vendor_tax_id"] = conf_vendor

        # 3. Extract PO Reference
        po_reference, conf_po = self._extract_po_reference(raw_text, base_confidence)
        field_confidences["po_reference"] = conf_po

        # 4. Extract Amounts
        net_amount, total_amount, conf_amount = self._extract_amounts(raw_text, base_confidence)
        field_confidences["amounts"] = conf_amount

        # 5. Extract Bank IBAN / Routing
        iban, conf_iban = self._extract_bank_routing(raw_text, base_confidence)
        field_confidences["bank_iban"] = conf_iban

        # 6. Extract Line Items
        line_items, conf_lines = self._extract_line_items(raw_text, base_confidence)
        field_confidences["line_items"] = conf_lines

        # Calculate weighted overall confidence Ce
        weights = {
            "invoice_id": 0.20,
            "vendor_tax_id": 0.20,
            "po_reference": 0.10,
            "amounts": 0.25,
            "bank_iban": 0.15,
            "line_items": 0.10,
        }
        overall_confidence = sum(field_confidences.get(k, 0.5) * w for k, w in weights.items())
        overall_confidence = round(min(1.0, max(0.0, overall_confidence)), 4)

        # Route to async validation queue if below threshold
        routed_to_queue = overall_confidence < self.confidence_threshold
        if routed_to_queue:
            logger.warn(
                "ocr.low_confidence_routed",
                invoice_id=invoice_id,
                overall_confidence=overall_confidence,
                threshold=self.confidence_threshold
            )

        invoice = Invoice(
            invoice_id=invoice_id or f"INV-UNKNOWN-{int(datetime.utcnow().timestamp())}",
            vendor_id=vendor_id or "VEND-UNKNOWN",
            po_reference=po_reference,
            invoice_date=datetime.utcnow(),
            net_amount=net_amount,
            total_amount=total_amount,
            bank_iban=iban,
            vendor_tax_id=tax_id,
            line_items=line_items,
            extraction_confidence=overall_confidence,
            raw_document_uri=document_uri,
        )

        result = OCRProcessingResult(
            invoice=invoice,
            overall_confidence=overall_confidence,
            routed_to_validation_queue=routed_to_queue,
            field_confidences=field_confidences,
            raw_text=raw_text,
            errors=errors,
        )

        if routed_to_queue:
            self.async_validation_queue.append(result)

        return result

    def _extract_invoice_id(self, text: str, base_conf: float) -> Tuple[Optional[str], float]:
        pattern = re.compile(
            r'(?:Invoice(?:\s*(?:#|Number|ID|No\.?))?[:\s]*)(INV-[A-Z0-9_-]+|[A-Z0-9_-]+)|\b(INV-[A-Z0-9_-]+)',
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            val = match.group(1) or match.group(2)
            if val and val.lower() != "invoice":
                return val.strip(), base_conf
        return None, 0.40

    def _extract_vendor_identifiers(self, text: str, base_conf: float) -> Tuple[Optional[str], Optional[str], float]:
        vend_match = re.search(r"(?:Vendor\s*ID|Supplier\s*ID|VEND[-:\s]*)\s*([A-Z0-9_-]+)", text, re.IGNORECASE)
        tax_match = re.search(r"(?:Tax\s*ID|EIN|VAT[-:\s]*)\s*([0-9A-Z-]+)", text, re.IGNORECASE)

        vendor_id = vend_match.group(1).strip() if vend_match else None
        tax_id = tax_match.group(1).strip() if tax_match else None

        if vendor_id and tax_id:
            return vendor_id, tax_id, base_conf
        if vendor_id or tax_id:
            return vendor_id or "VEND-UNRESOLVED", tax_id, base_conf * 0.8
        return None, None, 0.35

    def _extract_po_reference(self, text: str, base_conf: float) -> Tuple[Optional[str], float]:
        match = re.search(r"(?:PO\s*#?|Purchase\s*Order[-:\s]*)\s*([A-Z0-9_-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), base_conf
        return None, 0.60  # PO is not always on direct spot invoices

    def _extract_amounts(self, text: str, base_conf: float) -> Tuple[float, float, float]:
        total_match = re.search(
            r"(?:Total(?:\s*Amount)?|Grand\s*Total|Balance\s*Due)[-:\s]*\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE
        )
        if total_match:
            try:
                raw_amt = total_match.group(1).replace(",", "")
                val = float(raw_amt)
                return val, val, base_conf
            except ValueError:
                pass
        return 0.0, 0.0, 0.30

    def _extract_bank_routing(self, text: str, base_conf: float) -> Tuple[Optional[str], float]:
        match = re.search(r"(?:IBAN|Bank\s*Account|Routing)[-:\s]*([A-Z0-9]{8,34})", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), base_conf
        return None, 0.50

    def _extract_line_items(self, text: str, base_conf: float) -> Tuple[List[InvoiceLineItem], float]:
        # Basic line items extraction pattern
        lines = []
        pattern = re.compile(r"Item\s*(\d+)[:\s]+(.*?)\s+Qty[:\s]+(\d+)\s+Price[:\s]+\$?([\d,]+\.?\d*)", re.IGNORECASE)
        for match in pattern.finditer(text):
            try:
                line_no = int(match.group(1))
                desc = match.group(2).strip()
                qty = float(match.group(3))
                price = float(match.group(4).replace(",", ""))
                lines.append(
                    InvoiceLineItem(
                        line_number=line_no,
                        item_description=desc,
                        quantity=qty,
                        unit_price=price,
                        total_amount=qty * price
                    )
                )
            except Exception:
                continue
        return lines, base_conf if lines else 0.70
