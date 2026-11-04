"""Training + evaluation pipeline (runs locally, or as a Vertex AI custom job — same entrypoint).

  python -m p2p.ml.train                      # reads $DATA_DIR/warehouse.sqlite, writes $DATA_DIR/models/<version>/

Protocol (honest by construction):
  * temporal split — first 65% of invoices train, next 15% calibrate, last 20% test (never shuffled)
  * Tier 2 is trained/evaluated on the population that *passes Tier 1*, i.e. what it will see in production
  * end-to-end numbers (Tier 1 ∪ Tier 2) are reported on the full test period
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from p2p.common.config import get_settings
from p2p.data.repository import LabelStore, Warehouse
from p2p.features.builder import FEATURE_ORDER, FeatureBuilder
from p2p.features.graph import KnowledgeGraph
from p2p.ml.drift import make_baseline
from p2p.ml.scorer import Bundle, Tier2Scorer, classify, composite, graph_score
from p2p.rules.engine import RuleEngine


def build_frame(wh: Warehouse, labels: dict[str, tuple[int, str]]) -> pd.DataFrame:
    fb, rules = FeatureBuilder(wh, KnowledgeGraph.from_warehouse(wh)), RuleEngine()
    rows = []
    for iid in wh.invoice_ids():
        tx = fb.load(iid)
        t1 = rules.evaluate(tx)
        rows.append({"invoice_id": iid, "vendor_id": tx.invoice.vendor_id, "ts": tx.invoice.invoice_date, "amount": tx.invoice.total_amount,
                     "is_fraud": labels[iid][0], "fraud_type": labels[iid][1], "t1_blocked": int(t1.is_blocked),
                     "t1_rules": "|".join(t1.violations), **tx.features})
    return pd.DataFrame(rows).sort_values(["ts", "invoice_id"]).reset_index(drop=True)


def p_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    return float(y[np.argsort(-s)[:k]].mean()) if k else 0.0


def train(wh_path: Path, out_dir: Path, seed: int = 11) -> dict:
    t0 = time.time()
    wh, labels = Warehouse(wh_path, readonly=True), LabelStore(wh_path, readonly=True).labels()
    df = build_frame(wh, labels)
    # Warm-up: the first 90 days of the synthetic world have no prior history (cold-start artefact, not real behaviour).
    # They remain *history* for later invoices but are excluded from training/evaluation.
    df = df[df.ts >= df.ts.min() + pd.Timedelta(days=90)].reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(0.65 * n), int(0.80 * n)
    tr, va, te = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    s = get_settings()

    # Tier-2 population = rows that pass Tier 1
    tr2, va2, te2 = (d[d.t1_blocked == 0] for d in (tr, va, te))
    Xtr, ytr = tr2[FEATURE_ORDER].values, tr2.is_fraud.values
    pos, neg = int(ytr.sum()), int((1 - ytr).sum())
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=min(20.0, neg / max(pos, 1)), random_state=seed,
                             verbose=-1)
    clf.fit(Xtr, ytr)

    # Platt calibration on the validation slice
    raw_va = clf.predict_proba(va2[FEATURE_ORDER].values)[:, 1]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))  # noqa: E731
    if va2.is_fraud.nunique() == 2:
        lr = LogisticRegression(C=10.0).fit(logit(raw_va).reshape(-1, 1), va2.is_fraud.values)
        platt = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
    else:  # tiny datasets: a one-class calibration slice cannot be fitted; keep the raw (monotone) probabilities
        platt = (1.0, 0.0)

    # Unsupervised model: labels never used (cold-start / zero-day coverage)
    scaler = StandardScaler().fit(tr[FEATURE_ORDER].values)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(scaler.transform(tr[FEATURE_ORDER].values))
    anom_ref = np.sort(-iso.score_samples(scaler.transform(tr[FEATURE_ORDER].values)))

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + hashlib.sha1(df[FEATURE_ORDER].values.tobytes()).hexdigest()[:7]
    baseline = {f: make_baseline(tr[f].values) for f in FEATURE_ORDER}
    bundle = Bundle(version, clf, platt, scaler, iso, anom_ref, FEATURE_ORDER, baseline)
    sc = Tier2Scorer(bundle)

    # ── evaluation on the held-out future ────────────────────────────────────
    def tier2_scores(d: pd.DataFrame) -> np.ndarray:
        X = d[FEATURE_ORDER].values
        sup, an = sc.supervised_proba(X), sc.anomaly_tail(X)
        g = np.array([graph_score(r) for r in d[FEATURE_ORDER].to_dict("records")])
        return np.array([composite(a, b, c) for a, b, c in zip(sup, an, g)]), sup

    te2 = te2.copy()
    te2["score"], te2["sup"] = tier2_scores(te2)
    y2 = te2.is_fraud.values
    m: dict = {"tier2_population": {"n": int(len(te2)), "fraud": int(y2.sum()),
        "auc_pr_composite": round(float(average_precision_score(y2, te2.score)), 4) if y2.sum() else None,
        "auc_pr_supervised_only": round(float(average_precision_score(y2, te2.sup)), 4) if y2.sum() else None,
        "auc_roc": round(float(roc_auc_score(y2, te2.score)), 4) if 0 < y2.sum() < len(y2) else None,
        "precision_at_10": round(p_at_k(y2, te2.score.values, 10), 3), "precision_at_25": round(p_at_k(y2, te2.score.values, 25), 3),
        "brier_supervised_calibrated": round(float(brier_score_loss(y2, te2.sup)), 5),
        "base_rate": round(float(y2.mean()), 4)}}

    te = te.copy()
    te["score"] = 0.0
    te.loc[te.t1_blocked == 0, "score"] = te2.score.values
    te["flagged"] = ((te.t1_blocked == 1) | (te.score >= s.TAU_1)).astype(int)
    legit, fraud = te[te.is_fraud == 0], te[te.is_fraud == 1]
    per_type = {k: {"n": int(len(g)), "caught_by_tier1": int(g.t1_blocked.sum()), "caught_by_tier2_only": int(((g.t1_blocked == 0) & (g.score >= s.TAU_1)).sum()),
                    "recall": round(float(g.flagged.mean()), 3)} for k, g in fraud.groupby("fraud_type")}
    m["end_to_end_test"] = {
        "n": int(len(te)), "fraud": int(len(fraud)), "recall_overall": round(float(fraud.flagged.mean()), 3),
        "flag_rate_legit(FPR)": round(float(legit.flagged.mean()), 4), "tier1_fpr": round(float(legit.t1_blocked.mean()), 4),
        "precision_of_flags": round(float(te[te.flagged == 1].is_fraud.mean()), 3),
        "auto_approve_rate": round(float((te.flagged == 0).mean()), 3),
        "fraud_dollars_flagged_pct": round(float(fraud[fraud.flagged == 1].amount.sum() / max(fraud.amount.sum(), 1.0)), 3),
        "thresholds": {"tau1": s.TAU_1, "tau2": s.TAU_2}, "per_fraud_type": per_type}
    imp = sorted(zip(FEATURE_ORDER, clf.booster_.feature_importance("gain")), key=lambda x: -x[1])[:10]
    card = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "train_seconds": round(time.time() - t0, 1),
            "data": {"invoices": n, "train": len(tr), "calibration": len(va), "test": len(te), "tier2_train_pos": pos, "tier2_train_neg": neg,
                     "synthetic": True, "split": "temporal 65/15/20"},
            "algorithm": {"supervised": "LightGBM (scale_pos_weight, Platt-calibrated)", "unsupervised": "IsolationForest (top-5% tail)",
                          "combination": "noisy-OR with graph-link score", "explainability": "TreeSHAP via pred_contrib"},
            "feature_importance_gain": {k: round(float(v), 1) for k, v in imp}, "metrics": m,
            "caveats": ["Trained on synthetic data: absolute numbers are optimistic and only demonstrate the pipeline.",
                        "Tier-2 evaluated on rule-passing invoices only."]}
    ver_dir = out_dir / version
    ver_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ver_dir / "bundle.joblib")
    (ver_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    (out_dir / "current").write_text(version)
    df.drop(columns=["ts"]).assign(ts=df.ts.astype(str)).to_csv(out_dir.parent / "features.csv", index=False)
    return card


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--warehouse", type=Path, default=s.warehouse_path)
    ap.add_argument("--out", type=Path, default=s.model_dir)
    a = ap.parse_args()
    card = train(a.warehouse, a.out)
    print(json.dumps({"version": card["version"], **card["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
