"""ML drift monitoring (spec §8.1): PSI + Wasserstein per feature against the training baseline stored in the model bundle.

  PSI < 0.10 stable · 0.10–0.25 marginal (schedule shadow retrain) · ≥ 0.25 severe (alert, fall back to rules-heavy policy)
In production run daily as a Cloud Run job reading the last N days of features from BigQuery and publishing to Cloud Monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance


def make_baseline(sample: np.ndarray) -> dict:
    """Training-time summary stored in the model bundle: bin edges + *expected proportions measured on the real sample*.
    (Re-deriving proportions from quantile knots is ill-defined for heavily tied/capped features.)"""
    sample = np.asarray(sample, dtype=float)
    q = np.quantile(sample, np.linspace(0, 1, 101))
    inner = np.unique(q)[1:-1]
    counts = np.histogram(sample, bins=np.concatenate([[-np.inf], inner, [np.inf]]))[0]
    return {"quantiles": np.round(q, 4).tolist(), "edges": inner.tolist(), "expected": (counts / counts.sum()).tolist()}


def psi(baseline: dict, actual: np.ndarray, eps: float = 1e-4) -> float:
    """PSI = Σ (A% − E%)·ln(A%/E%) over the stored bins."""
    actual = np.asarray(actual, dtype=float)
    if len(actual) == 0:
        return 0.0
    counts = np.histogram(actual, bins=np.concatenate([[-np.inf], baseline["edges"], [np.inf]]))[0]
    e = np.clip(np.asarray(baseline["expected"]), eps, None)
    a = np.clip(counts / counts.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def severity(value: float) -> str:
    return "STABLE" if value < 0.10 else ("MARGINAL_SHIFT" if value < 0.25 else "SEVERE_DRIFT")


@dataclass
class DriftReport:
    per_feature: dict[str, dict]
    worst_feature: str
    max_psi: float
    overall: str

    @property
    def needs_shadow_retrain(self) -> bool:
        return self.overall in ("MARGINAL_SHIFT", "SEVERE_DRIFT")


def drift_report(baseline: dict[str, dict], recent: dict[str, np.ndarray]) -> DriftReport:
    rows = {}
    for f, q in baseline.items():
        if f not in recent or len(recent[f]) < 20:
            continue
        v = psi(q, np.asarray(recent[f], dtype=float))
        rows[f] = {"psi": round(v, 4), "severity": severity(v), "wasserstein": round(float(wasserstein_distance(q["quantiles"], recent[f])), 4)}
    worst = max(rows, key=lambda k: rows[k]["psi"]) if rows else ""
    mx = rows[worst]["psi"] if rows else 0.0
    return DriftReport(rows, worst, mx, severity(mx))
