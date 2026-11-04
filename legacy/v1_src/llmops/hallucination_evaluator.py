"""Automated Factual Consistency Evaluator checking agent narratives against ground-truth facts."""
import re
from typing import Any, Dict, List, Optional
import structlog
from pydantic import BaseModel
from src.core.models import CaseDossier, Invoice, Vendor

logger = structlog.get_logger(__name__)


class HallucinationEvaluationReport(BaseModel):
    case_id: str
    is_factually_consistent: bool
    hallucination_detected: bool
    ungrounded_claims: List[str]
    factual_accuracy_score: float  # [0.0, 1.0]


class FactualHallucinationEvaluator:
    """Evaluates agent-generated narratives against ground truth ERP and DB facts.
    
    Mitigates hallucination drift and ensures Factual Hallucination Rate (FHR) < 0.1%.
    """

    def evaluate(
        self,
        dossier: CaseDossier,
        source_invoice: Invoice,
        source_vendor: Optional[Vendor] = None
    ) -> HallucinationEvaluationReport:
        """Verify that claims in agent_narrative correctly match source records."""
        narrative = dossier.agent_narrative
        ungrounded: List[str] = []

        # 1. Check Invoice ID grounding
        if source_invoice.invoice_id not in narrative:
            ungrounded.append(f"Narrative does not contain true invoice ID: {source_invoice.invoice_id}")

        # 2. Check Vendor ID grounding
        if source_invoice.vendor_id not in narrative:
            ungrounded.append(f"Narrative does not contain true vendor ID: {source_invoice.vendor_id}")

        # 3. Check Amount Grounding
        # Search for dollar amounts in narrative
        amt_matches = re.findall(r"\$([\d,]+\.?\d*)", narrative)
        if amt_matches:
            found_correct_amount = False
            for raw in amt_matches:
                try:
                    val = float(raw.replace(",", ""))
                    if abs(val - source_invoice.total_amount) < 0.05:
                        found_correct_amount = True
                        break
                except ValueError:
                    pass
            if not found_correct_amount:
                ungrounded.append(
                    f"Narrative amounts ({amt_matches}) do not match invoice total (${source_invoice.total_amount:,.2f})"
                )

        # 4. Check Evidence Manifest Non-Emptiness
        if not dossier.evidence_manifest:
            ungrounded.append("Evidence manifest is empty; narrative is ungrounded")

        has_hallucinations = len(ungrounded) > 0
        accuracy = 1.0 - (len(ungrounded) * 0.25)
        accuracy = max(0.0, min(1.0, accuracy))

        if has_hallucinations:
            logger.warn(
                "llmops.hallucination_detected",
                case_id=dossier.case_id,
                ungrounded_claims=ungrounded
            )

        return HallucinationEvaluationReport(
            case_id=dossier.case_id,
            is_factually_consistent=not has_hallucinations,
            hallucination_detected=has_hallucinations,
            ungrounded_claims=ungrounded,
            factual_accuracy_score=round(accuracy, 2),
        )
