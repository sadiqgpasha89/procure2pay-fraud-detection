"""Tier 2 — probabilistic scoring: LightGBM (supervised) + Isolation Forest (cold-start / novel) + graph.

Composite risk uses a noisy-OR so each independent signal can only *add* evidence:
    S = 1 − (1−p_sup)·(1−0.5·a)·(1−0.7·g)  (+ circuit-breaker fallback penalty), clipped to [0,1]
where p_sup is the calibrated GBDT probability, a the tail-rank of the Isolation Forest score, g the graph-link score.
Local explanations are exact TreeSHAP values from LightGBM (`pred_contrib=True`) — no extra dependency.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from p2p.common.config import get_settings
from p2p.common.models import RiskClass, ShapContribution, Tier2Result
from p2p.features.builder import DISPLAY
from p2p.ml.fast_iforest import FastIForest


def graph_score(f: dict[str, float]) -> float:
    s = 0.0
    s += 0.7 * f.get("staff_vendor_link_flag", 0.0)
    s += 0.4 * f.get("vendor_employee_link_flag", 0.0)
    s += 0.5 * f.get("vendor_shared_bank_flag", 0.0)
    s += 0.3 * f.get("vendor_shared_address_flag", 0.0)
    return float(min(1.0, s))


def composite(sup: float, anom: float, graph: float, penalty: float = 0.0) -> float:
    s = 1.0 - (1.0 - sup) * (1.0 - 0.5 * anom) * (1.0 - 0.7 * graph)
    return float(min(1.0, max(0.0, s + penalty)))


def classify(score: float, tau1: float, tau2: float) -> RiskClass:
    return RiskClass.LOW if score < tau1 else (RiskClass.MEDIUM if score < tau2 else RiskClass.HIGH)


@dataclass
class Bundle:
    version: str
    lgbm: object  # lightgbm.LGBMClassifier
    platt: tuple[float, float]
    scaler: object
    iforest: object
    anom_ref: np.ndarray  # sorted train-set anomaly raw scores → percentile rank
    feature_order: list[str]
    baseline: dict[str, dict]  # per-feature training distribution: edges/expected/quantiles (drift monitor)


class Tier2Scorer:
    def __init__(self, bundle: Bundle):
        self.b = bundle
        self.s = get_settings()
        self._fast_if = FastIForest(bundle.iforest)
        bundle.lgbm.set_params(n_jobs=1)  # single-row inference: OpenMP fan-out costs more than the work and thrashes under concurrency

    @classmethod
    def load(cls, model_dir: Path | None = None, version: str | None = None) -> "Tier2Scorer":
        root = model_dir or get_settings().model_dir
        version = version or (root / "current").read_text().strip()
        return cls(joblib.load(root / version / "bundle.joblib"))

    @staticmethod
    def model_card(model_dir: Path | None = None) -> dict:
        root = model_dir or get_settings().model_dir
        version = (root / "current").read_text().strip()
        return json.loads((root / version / "model_card.json").read_text())

    # ── vectorised helpers (also used by training/eval) ─────────────────────
    def matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.array([[r.get(k, 0.0) for k in self.b.feature_order] for r in rows], dtype=float)

    def supervised_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.b.lgbm.predict_proba(X)[:, 1]
        a, b = self.b.platt
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return 1.0 / (1.0 + np.exp(-(a * logit + b)))

    def anomaly_tail(self, X: np.ndarray) -> np.ndarray:
        raw = -self._fast_if.score_samples(self.b.scaler.transform(X))  # higher = more anomalous
        pct = np.searchsorted(self.b.anom_ref, raw) / len(self.b.anom_ref)
        return np.clip((pct - 0.95) / 0.05, 0.0, 1.0)  # only the top 5% tail counts as anomalous

    # ── online scoring ──────────────────────────────────────────────────────
    def score(self, features: dict[str, float], fallback_penalty: float = 0.0, top_k: int = 5) -> Tier2Result:
        t0 = time.perf_counter()
        X = self.matrix([features])
        sup = float(self.supervised_proba(X)[0])
        anom = float(self.anomaly_tail(X)[0])
        g = graph_score(features)
        score = round(composite(sup, anom, g, fallback_penalty), 4)
        contrib = self.b.lgbm.booster_.predict(X, pred_contrib=True, num_threads=1)[0][:-1]
        order = np.argsort(-contrib)[:top_k]
        factors = [ShapContribution(feature=self.b.feature_order[i], value=round(float(X[0, i]), 4), shap_value=round(float(contrib[i]), 4))
                   for i in order if contrib[i] > 0]
        return Tier2Result(composite_score=score, supervised_score=round(sup, 4), anomaly_score=round(anom, 4), graph_score=round(g, 4),
                           classification=classify(score, self.s.TAU_1, self.s.TAU_2), contributions=factors,
                           fallback_penalty=fallback_penalty, model_version=self.b.version,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 3))


def describe(feature: str) -> str:
    return DISPLAY.get(feature, feature.replace("_", " "))

