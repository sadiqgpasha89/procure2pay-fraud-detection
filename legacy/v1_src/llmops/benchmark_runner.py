"""LLMOps Benchmark Runner evaluating the 50 synthetic historical fraud scenarios."""
from datetime import datetime
from typing import Any, Dict, List
import structlog
from pydantic import BaseModel
from src.core.models import Invoice, RecommendedAction, Vendor
from src.detection.pipeline import TriTierDetectionPipeline
from src.llmops.hallucination_evaluator import FactualHallucinationEvaluator

logger = structlog.get_logger(__name__)


class BenchmarkScenario(BaseModel):
    scenario_id: str
    scenario_type: str
    invoice: Invoice
    vendor: Vendor
    expected_action: RecommendedAction


class BenchmarkResult(BaseModel):
    total_scenarios: int
    passed_scenarios: int
    pass_rate: float
    hallucination_count: int
    average_latency_ms: float
    failures: List[Dict[str, Any]]


class LLMOpsBenchmarkRunner:
    """Continuous evaluation runner for 50 historical fraud benchmark test cases."""

    def __init__(self, pipeline: TriTierDetectionPipeline):
        self.pipeline = pipeline
        self.hallucination_evaluator = FactualHallucinationEvaluator()

    def generate_benchmark_suite(self, count: int = 50) -> List[BenchmarkScenario]:
        """Generate the benchmark dataset of 50 canonical fraud and edge scenarios."""
        scenarios: List[BenchmarkScenario] = []

        for i in range(count):
            idx = i + 1
            if i % 4 == 0:
                # Bank changed <72h (Immediate block / hold)
                v = Vendor(
                    vendor_id=f"VEND-BENCH-{idx}",
                    legal_name=f"Alpha Supply Corp {idx}",
                    tax_id=f"TAX-100{idx}",
                    bank_iban=f"US89BANK{idx:08d}",
                    bank_account_updated_at=datetime.utcnow(),  # Fresh bank update
                    address=f"{idx} Industrial Blvd, New York, NY",
                )
                inv = Invoice(
                    invoice_id=f"INV-BENCH-{idx}",
                    vendor_id=v.vendor_id,
                    total_amount=48500.00,
                    net_amount=48500.00,
                )
                exp = RecommendedAction.BLOCK_TRANSACTION
                s_type = "BANK_CHANGE_72H"

            elif i % 4 == 1:
                # Near threshold smurfing invoice
                v = Vendor(
                    vendor_id=f"VEND-BENCH-{idx}",
                    legal_name=f"Beta Logistics {idx}",
                    tax_id=f"TAX-200{idx}",
                    bank_iban=f"US89BANK{idx:08d}",
                    address=f"{idx} Harbor Rd, Newark, NJ",
                )
                inv = Invoice(
                    invoice_id=f"INV-BENCH-{idx}",
                    vendor_id=v.vendor_id,
                    total_amount=49950.00,  # Just below $50k
                    net_amount=49950.00,
                )
                exp = RecommendedAction.REQUEST_MANUAL_KYC
                s_type = "NEAR_THRESHOLD_SMURFING"

            elif i % 4 == 2:
                # Sanctioned entity
                v = Vendor(
                    vendor_id=f"VEND-BENCH-{idx}",
                    legal_name=f"Gamma Holdings {idx}",
                    tax_id=f"TAX-300{idx}",
                    bank_iban=f"US89BANK{idx:08d}",
                    address=f"{idx} Offshore Ave, Panama",
                    is_sanctioned=True,
                )
                inv = Invoice(
                    invoice_id=f"INV-BENCH-{idx}",
                    vendor_id=v.vendor_id,
                    total_amount=15000.00,
                    net_amount=15000.00,
                )
                exp = RecommendedAction.BLOCK_TRANSACTION
                s_type = "SANCTIONED_VENDOR"

            else:
                # Low risk normal transaction
                v = Vendor(
                    vendor_id=f"VEND-BENCH-{idx}",
                    legal_name=f"Reliable Parts Inc {idx}",
                    tax_id=f"TAX-400{idx}",
                    bank_iban=f"US89BANK{idx:08d}",
                    address=f"{idx} Main St, Chicago, IL",
                )
                inv = Invoice(
                    invoice_id=f"INV-BENCH-{idx}",
                    vendor_id=v.vendor_id,
                    total_amount=1200.00,
                    net_amount=1200.00,
                )
                exp = RecommendedAction.AUTO_APPROVE
                s_type = "CLEAN_TRANSACTION"

            scenarios.append(
                BenchmarkScenario(
                    scenario_id=f"BENCH-SCENARIO-{idx:03d}",
                    scenario_type=s_type,
                    invoice=inv,
                    vendor=v,
                    expected_action=exp,
                )
            )

        return scenarios

    def run_benchmark(self, count: int = 50) -> BenchmarkResult:
        """Execute evaluation across the benchmark suite."""
        suite = self.generate_benchmark_suite(count)
        passed = 0
        hallucinations = 0
        latencies: List[float] = []
        failures: List[Dict[str, Any]] = []

        logger.info("llmops.benchmark_started", total_cases=count)

        for s in suite:
            outcome = self.pipeline.evaluate_transaction(
                invoice=s.invoice,
                vendor=s.vendor,
            )
            latencies.append(outcome.total_latency_ms)

            # Check action match
            is_match = (outcome.final_action == s.expected_action)
            # If agentic dossier generated, verify factual grounding
            if outcome.case_dossier:
                h_report = self.hallucination_evaluator.evaluate(
                    outcome.case_dossier,
                    s.invoice,
                    s.vendor
                )
                if h_report.hallucination_detected:
                    hallucinations += 1
                    is_match = False

            if is_match:
                passed += 1
            else:
                failures.append({
                    "scenario_id": s.scenario_id,
                    "expected": s.expected_action.value,
                    "actual": outcome.final_action.value,
                    "type": s.scenario_type,
                })

        pass_rate = round(passed / max(1, count), 4)
        avg_lat = round(sum(latencies) / max(1, len(latencies)), 2)

        logger.info(
            "llmops.benchmark_completed",
            pass_rate=pass_rate,
            hallucinations=hallucinations,
            avg_latency_ms=avg_lat
        )

        return BenchmarkResult(
            total_scenarios=count,
            passed_scenarios=passed,
            pass_rate=pass_rate,
            hallucination_count=hallucinations,
            average_latency_ms=avg_lat,
            failures=failures,
        )
