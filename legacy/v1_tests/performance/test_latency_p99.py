"""Performance benchmark validating P99 <= 850ms latency SLA (Section 7.4)."""
import time
import numpy as np
import pytest
from src.core.models import Invoice, Vendor
from src.core.telemetry import MetricsTracker
from src.detection.pipeline import TriTierDetectionPipeline


@pytest.mark.performance
def test_p99_latency_sla_under_850ms(tri_tier_pipeline, clean_invoice, clean_vendor, sample_po):
    """Benchmark end-to-end transaction evaluation latency and assert P99 <= 850ms."""
    tracker = MetricsTracker()
    n_iterations = 100

    latencies_ms = []

    # Warm-up run
    tri_tier_pipeline.evaluate_transaction(clean_invoice, clean_vendor, sample_po)

    for i in range(n_iterations):
        inv = clean_invoice.model_copy(update={"invoice_id": f"INV-PERF-{i:04d}"})
        start = time.perf_counter()
        outcome = tri_tier_pipeline.evaluate_transaction(
            invoice=inv,
            vendor=clean_vendor,
            po=sample_po,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(duration_ms)
        tracker.record_latency("decision_latency", duration_ms)

    stats = tracker.get_latency_stats("decision_latency")

    print(f"\n--- Performance Benchmark Results (N={n_iterations}) ---")
    print(f"Mean Latency: {stats['mean_ms']:.2f} ms")
    print(f"P50 Latency:  {stats['p50_ms']:.2f} ms")
    print(f"P90 Latency:  {stats['p90_ms']:.2f} ms")
    print(f"P95 Latency:  {stats['p95_ms']:.2f} ms")
    print(f"P99 Latency:  {stats['p99_ms']:.2f} ms")
    print(f"Max Latency:  {stats['max_ms']:.2f} ms")

    # Strict SLA Assertion: P99 must be well within the <= 850ms target
    assert stats["p99_ms"] <= 850.0, f"P99 latency of {stats['p99_ms']}ms exceeded SLA of 850ms!"
