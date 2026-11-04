"""Adapters normalizing heterogeneous ERP payloads (SAP S/4HANA, Oracle NetSuite) into canonical schema."""
from datetime import datetime
from typing import Any, Dict
from src.core.models import Invoice, InvoiceLineItem, PurchaseOrder, GoodsReceipt


class SAPS4HanaAdapter:
    """Adapter for SAP S/4HANA IDoc / OData API payloads."""

    @staticmethod
    def parse_invoice(payload: Dict[str, Any]) -> Invoice:
        # Example SAP fields: BELNR (Doc No), LIFNR (Vendor), WRBTR (Amount), WAERS (Currency)
        inv_id = payload.get("BELNR") or payload.get("InvoiceNumber", "")
        vendor_id = payload.get("LIFNR") or payload.get("VendorNumber", "")
        po_ref = payload.get("EBELN") or payload.get("PurchaseOrderNumber")
        gross_amount = float(payload.get("WRBTR", payload.get("GrossAmount", 0.0)))
        tax_amount = float(payload.get("WMWST", payload.get("TaxAmount", 0.0)))
        currency = payload.get("WAERS", "USD")
        iban = payload.get("IBAN")

        raw_items = payload.get("ITEMS", [])
        line_items = []
        for idx, it in enumerate(raw_items, start=1):
            line_items.append(
                InvoiceLineItem(
                    line_number=idx,
                    item_description=it.get("TXZ01", it.get("Description", "Line item")),
                    quantity=float(it.get("MENGE", 1.0)),
                    unit_price=float(it.get("NETPR", 0.0)),
                    total_amount=float(it.get("NETWR", 0.0)),
                )
            )

        return Invoice(
            invoice_id=str(inv_id),
            vendor_id=str(vendor_id),
            po_reference=str(po_ref) if po_ref else None,
            invoice_date=datetime.utcnow(),
            net_amount=gross_amount - tax_amount,
            tax_amount=tax_amount,
            total_amount=gross_amount,
            currency=currency,
            bank_iban=iban,
            line_items=line_items,
            entity_code=payload.get("BUKRS", "ENT-01"),
        )


class NetSuiteAdapter:
    """Adapter for Oracle NetSuite SuiteTalk / REST Web Services payloads."""

    @staticmethod
    def parse_invoice(payload: Dict[str, Any]) -> Invoice:
        # Example NetSuite fields: tranId, entity (vendor), total, subtotal, currency
        inv_id = payload.get("tranId") or payload.get("id", "")
        vendor_obj = payload.get("entity", {})
        vendor_id = vendor_obj.get("id") if isinstance(vendor_obj, dict) else str(vendor_obj)
        po_ref = payload.get("createdFrom", {}).get("id") if isinstance(payload.get("createdFrom"), dict) else payload.get("createdFrom")
        total = float(payload.get("total", 0.0))
        subtotal = float(payload.get("subtotal", total))

        return Invoice(
            invoice_id=str(inv_id),
            vendor_id=str(vendor_id),
            po_reference=str(po_ref) if po_ref else None,
            invoice_date=datetime.utcnow(),
            net_amount=subtotal,
            tax_amount=total - subtotal,
            total_amount=total,
            currency=payload.get("currencyName", "USD"),
            entity_code=str(payload.get("subsidiary", "ENT-01")),
        )


class ERPAdapterFactory:
    """Factory creating appropriate ERP adapter based on system type."""

    @staticmethod
    def parse(erp_type: str, payload: Dict[str, Any]) -> Invoice:
        erp_type_norm = erp_type.lower()
        if "sap" in erp_type_norm:
            return SAPS4HanaAdapter.parse_invoice(payload)
        elif "netsuite" in erp_type_norm:
            return NetSuiteAdapter.parse_invoice(payload)
        else:
            # Direct canonical fallback
            return Invoice(**payload)
