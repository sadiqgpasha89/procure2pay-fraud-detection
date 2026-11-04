"""Telemetry, structured logging, and P99 latency tracking."""
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class MetricsTracker:
    """In-memory rolling metrics tracker for P50, P95, and P99 latencies."""

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.latencies: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}

    def record_latency(self, metric_name: str, duration_ms: float) -> None:
        """Record latency duration in milliseconds."""
        if metric_name not in self.latencies:
            self.latencies[metric_name] = []
        bucket = self.latencies[metric_name]
        bucket.append(duration_ms)
        if len(bucket) > self.max_samples:
            bucket.pop(0)

    def increment_counter(self, metric_name: str, value: int = 1) -> None:
        """Increment count for a metric counter."""
        self.counters[metric_name] = self.counters.get(metric_name, 0) + value

    def get_latency_stats(self, metric_name: str) -> Dict[str, float]:
        """Compute P50, P90, P95, and P99 percentiles."""
        samples = self.latencies.get(metric_name, [])
        if not samples:
            return {"count": 0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        arr = np.array(samples)
        return {
            "count": len(samples),
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Return snapshot of all registered latencies and counters."""
        latency_snapshot = {name: self.get_latency_stats(name) for name in self.latencies}
        return {
            "latencies": latency_snapshot,
            "counters": self.counters,
        }

    def reset(self) -> None:
        """Clear recorded metrics."""
        self.latencies.clear()
        self.counters.clear()


# Global shared metrics tracker
global_metrics = MetricsTracker()


def time_execution(metric_name: str, tracker: Optional[MetricsTracker] = None):
    """Decorator to measure and record execution time of functions."""
    active_tracker = tracker or global_metrics

    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                active_tracker.record_latency(metric_name, elapsed_ms)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                active_tracker.record_latency(metric_name, elapsed_ms)

        import inspect
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator
