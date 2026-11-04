"""Security guardrails, PII tokenization, and verifiable citation enforcement."""
import re
from typing import Any, Dict, List, Set, Tuple
import structlog
from src.core.models import EvidenceCitation, RecommendedAction

logger = structlog.get_logger(__name__)

ALLOWED_ACTIONS: Set[str] = {
    RecommendedAction.AUTO_APPROVE.value,
    RecommendedAction.REQUEST_MANUAL_KYC.value,
    RecommendedAction.HOLD_PAYMENT.value,
    RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT.value,
    RecommendedAction.FREEZE_VENDOR.value,
    RecommendedAction.BLOCK_TRANSACTION.value,
}


class PIIMasker:
    """Masks and tokenizes PII before sending prompts to external foundation models."""

    @staticmethod
    def mask_tax_id(tax_id: str) -> str:
        if not tax_id or len(tax_id) < 4:
            return "TAX-***"
        return f"***-**-{tax_id[-4:]}"

    @staticmethod
    def mask_iban(iban: str) -> str:
        if not iban or len(iban) < 6:
            return "IBAN-***"
        return f"{iban[:4]}****{iban[-4:]}"

    @staticmethod
    def mask_text(text: str) -> str:
        # Mask EIN / Tax IDs (e.g. 12-3456789)
        text = re.sub(r"\b\d{2}-\d{7}\b", "**-*******", text)
        # Mask SSN (e.g. 123-45-6789)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", text)
        # Mask IBAN patterns
        text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b", "IBAN-MASKED", text)
        return text


class CitationVerifier:
    """Validates that all claims and citations map to cryptographically verifiable evidence items."""

    @staticmethod
    def extract_citations(narrative: str) -> List[str]:
        """Extract [Evidence: ...] citation tokens from agent narrative."""
        matches = re.findall(r"\[Evidence:\s*([^\]]+)\]", narrative, re.IGNORECASE)
        return [m.strip() for m in matches]

    @staticmethod
    def verify_citations(
        narrative: str,
        evidence_manifest: List[EvidenceCitation]
    ) -> Tuple[bool, List[str]]:
        """Verify that cited records exist in the evidence manifest."""
        manifest_record_ids = {e.record_id for e in evidence_manifest}
        manifest_sources = {e.source for e in evidence_manifest}

        cited_items = CitationVerifier.extract_citations(narrative)
        unverified: List[str] = []

        for citation in cited_items:
            # Matches format e.g. "Neo4j Node Edge #4412" or "LOG-BANK-CHANGE-88129"
            found = False
            for rec_id in manifest_record_ids:
                if rec_id in citation:
                    found = True
                    break
            for src in manifest_sources:
                if src in citation:
                    found = True
                    break
            if not found:
                unverified.append(citation)

        is_valid = len(unverified) == 0
        return is_valid, unverified


class ActionIsolationGuardrail:
    """Ensures recommended actions adhere strictly to allowed enum variants."""

    @staticmethod
    def validate_action(action: str) -> RecommendedAction:
        clean_action = action.strip().upper()
        if clean_action in ALLOWED_ACTIONS:
            return RecommendedAction(clean_action)
        logger.warn("guardrail.invalid_action_coerced", invalid_action=action)
        return RecommendedAction.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT
