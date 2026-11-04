"""Unit tests for LLMOps prompt registry, hallucination evaluator, and benchmark runner."""
import pytest
from src.core.models import (
    CaseDossier,
    EvidenceCitation,
    GraphFindings,
    Invoice,
    RecommendedAction,
    RiskAssessment,
    RiskClassification,
    TransactionIdentifiers,
    Vendor,
)
from src.llmops.benchmark_runner import LLMOpsBenchmarkRunner
from src.llmops.hallucination_evaluator import FactualHallucinationEvaluator
from src.llmops.metrics_tracker import LLMMetricsTracker
from src.llmops.prompt_registry import get_prompt_registry


@pytest.mark.unit
def test_prompt_registry():
    """Prompt registry should retrieve versioned prompt templates."""
    registry = get_prompt_registry()
    prompt = registry.get_prompt("risk_synthesizer", version="1.0.0")
    assert prompt is not None
    assert "Chief Enterprise Fraud Investigator" in prompt.system_prompt


@pytest.mark.unit
def test_factual_hallucination_evaluator(clean_invoice, clean_vendor):
    """Hallucination evaluator should verify true facts and catch fabricated claims."""
    evaluator = FactualHallucinationEvaluator()

    # 1. Factually consistent dossier
    consistent_dossier = CaseDossier(
        case_id="CASE-001",
        timestamp="2026-09-14T10:00:00Z",
        transaction_identifiers=TransactionIdentifiers(
            invoice_id=clean_invoice.invoice_id,
            vendor_id=clean_invoice.vendor_id,
        ),
        risk_assessment=RiskAssessment(
            composite_risk_score=0.2,
            classification=RiskClassification.LOW_RISK,
        ),
        graph_findings=GraphFindings(),
        agent_narrative=f"Invoice {clean_invoice.invoice_id} for vendor {clean_invoice.vendor_id} total ${clean_invoice.total_amount:,.2f}.",
        recommended_action=RecommendedAction.AUTO_APPROVE,
        requires_human_signoff=False,
        evidence_manifest=[EvidenceCitation(source="ERP", record_id="REC-1", verification_hash="hash1")],
    )
    report_good = evaluator.evaluate(consistent_dossier, clean_invoice, clean_vendor)
    assert report_good.is_factually_consistent is True
    assert report_good.hallucination_detected is False

    # 2. Fabricated / hallucinated dossier (wrong amount and wrong invoice ID)
    hallucinated_dossier = consistent_dossier.model_copy(
        update={"agent_narrative": "Invoice INV-FABRICATED for vendor VEND-FAKE total $999,999.00."}
    )
    report_bad = evaluator.evaluate(hallucinated_dossier, clean_invoice, clean_vendor)
    assert report_bad.is_factually_consistent is False
    assert report_bad.hallucination_detected is True
    assert len(report_bad.ungrounded_claims) > 0


@pytest.mark.unit
def test_llm_metrics_tracker():
    """LLMMetricsTracker should compute RAR and FHR properly."""
    tracker = LLMMetricsTracker()
    tracker.record_agent_dispatch(latency_sec=2.0)
    tracker.record_agent_dispatch(latency_sec=3.0)

    # 3 human approvals, 1 override
    tracker.record_human_decision(accepted=True)
    tracker.record_human_decision(accepted=True)
    tracker.record_human_decision(accepted=True)
    tracker.record_human_decision(accepted=False)

    metrics = tracker.get_metrics()
    assert metrics.total_dispatches == 2
    assert metrics.accepted_recommendations == 3
    assert metrics.recommendation_acceptance_rate == 0.75
    assert metrics.average_synthesis_latency_sec == 2.5


@pytest.mark.unit
def test_benchmark_runner_execution(tri_tier_pipeline):
    """Benchmark runner should evaluate scenarios and compute pass rate."""
    runner = LLMOpsBenchmarkRunner(pipeline=tri_tier_pipeline)
    # Run test on 8 scenarios
    results = runner.run_benchmark(count=8)

    assert results.total_scenarios == 8
    assert results.passed_scenarios >= 6
    assert results.pass_rate >= 0.75
    assert results.average_latency_ms > 0.0
