"""End-to-end benchmark against the RUNNING stack (spec §7 + §8.2): replays the held-out period through the gateway.

  python -m p2p.stack up && python -m p2p.llmops.benchmark --legit 80 --fraud 80 --out docs/EVAL_REPORT.md

Measures: triage precision/recall by fraud type · hot-path latency (P50/P95/P99) · investigation synthesis latency ·
dossier integrity (independent hash re-verification, guardrail interventions, tool failures) · prompt-injection handling ·
agent KPIs from simulated analyst dispositions · ledger integrity · ML drift on the test period.
Ground-truth labels are read here only (LabelStore) — agents and tools can never reach them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from p2p.common.config import get_settings
from p2p.common.evidence import content_hash
from p2p.data.repository import LabelStore, Warehouse
from p2p.ml.drift import drift_report
from p2p.ml.scorer import Tier2Scorer


def pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(xs, q)), 1) if xs else 0.0


def independent_verify(wh: Warehouse, record_id: str, h: str) -> bool | None:
    """Second, separate verification path (direct warehouse read, not via MCP). None = not an ERP record."""
    kind, _, key = record_id.partition(":")
    rec = {"INV": wh.get_invoice, "PO": wh.get_po, "VENDOR": wh.get_vendor}.get(kind)
    if rec is None:
        return None
    obj = rec(key)
    return obj is not None and content_hash(obj) == h


async def run(args: argparse.Namespace) -> dict:
    s = get_settings()
    wh, labels = Warehouse(s.warehouse_path, readonly=True), LabelStore(s.warehouse_path, readonly=True).labels()
    ids = wh.invoice_ids()
    feats = pd.read_csv(s.DATA_DIR / "features.csv")  # training-time frame (warm-up already removed, same temporal order)
    modelled = [i for i in feats.sort_values(["ts", "invoice_id"]).invoice_id]
    test_ids = modelled[int(0.8 * len(modelled)):]
    rng = random.Random(args.seed)
    fraud_ids = [i for i in test_ids if labels[i][0] == 1]
    legit_ids = [i for i in test_ids if labels[i][0] == 0]
    sample = rng.sample(fraud_ids, min(args.fraud, len(fraud_ids))) + rng.sample(legit_ids, min(args.legit, len(legit_ids)))
    # adversarial subset: vendors whose e-mail corpus contains an injection payload (any period)
    inj_vendors = {d.vendor_id for d in wh.documents("EMAIL") if "SYSTEM NOTICE" in d.text}
    adversarial = [i for i in ids if wh.get_invoice(i).vendor_id in inj_vendors and labels[i][1] == "BANK_DIVERSION"][:6]
    sample = list(dict.fromkeys(sample + adversarial))
    rng.shuffle(sample)

    out: dict = {"meta": {"llm_provider": s.LLM_PROVIDER, "sampled": len(sample), "fraud": sum(labels[i][0] for i in sample),
                          "legit": sum(1 - labels[i][0] for i in sample), "adversarial_injection_cases": len(adversarial)}}
    sem = asyncio.Semaphore(args.concurrency)
    decisions: dict[str, dict] = {}
    lat: dict[str, list[float]] = defaultdict(list)

    async with httpx.AsyncClient(timeout=120, headers=s.auth_headers()) as h:
        # Warm-up (not measured): first MCP session / A2A card resolution / model load per process are cold-start costs that
        # Cloud Run min-instances + startup probes absorb in production. Reported separately so it is not hidden.
        from p2p.common.models import InvestigationRequest, ScoringResult
        from p2p.services.gateway.investigator_client import InvestigatorClient

        warm = rng.sample([i for i in ids if i not in set(sample)], 3)
        t0 = time.perf_counter()
        for iid in warm:
            sc = ScoringResult.model_validate((await h.post(f"{s.SCORING_URL}/v1/score", json={"invoice_id": iid})).json())
            await InvestigatorClient().investigate(InvestigationRequest(case_id="CASE-WARMUP", invoice_id=iid, vendor_id=sc.vendor_id, amount=sc.amount,
                                                                        po_reference=sc.po_reference, scoring=sc))
            if iid == warm[0]:
                out["cold_start_first_investigation_s"] = round(time.perf_counter() - t0, 2)

        async def one(iid: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await h.post(f"{s.GATEWAY_URL}/v1/evaluate", json={"invoice_id": iid, "wait_for_investigation": False})
                wall = (time.perf_counter() - t0) * 1000
                r.raise_for_status()
                d = r.json()
                decisions[iid] = d
                lat[d["route"]].append(wall)
                lat["ALL"].append(wall)

        t0 = time.time()
        await asyncio.gather(*(one(i) for i in sample))
        out["hot_path_wall_seconds"] = round(time.time() - t0, 1)
        deadline = time.time() + args.investigation_timeout
        while time.time() < deadline:
            if (await h.get(f"{s.GATEWAY_URL}/v1/pending")).json()["background_investigations"] == 0:
                break
            await asyncio.sleep(1.0)
        cases = {c["invoice_id"]: c for c in (await h.get(f"{s.CASE_URL}/v1/cases?limit=1000")).json()}
        full = {iid: (await h.get(f"{s.CASE_URL}/v1/cases/{c['case_id']}")).json() for iid, c in cases.items() if iid in decisions}

        # ── triage quality ──
        tp = fp = fn = tn = 0
        by_type: dict[str, Counter] = defaultdict(Counter)
        for iid in sample:
            fraud, kind = labels[iid]
            esc = decisions[iid]["route"] != "APPROVE"
            tp += fraud and esc; fn += fraud and not esc; fp += (not fraud) and esc; tn += (not fraud) and not esc
            by_type[kind]["n"] += 1
            by_type[kind]["escalated"] += esc
            by_type[kind]["blocked_tier1"] += decisions[iid]["route"] == "BLOCK"
        out["triage"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
                         "false_positive_rate": round(fp / max(fp + tn, 1), 3), "note": "sample is enriched with fraud; FPR/recall are the portable numbers, precision is not"}
        out["by_fraud_type"] = {k: {"n": v["n"], "escalated": v["escalated"], "tier1_blocked": v["blocked_tier1"],
                                   "recall": round(v["escalated"] / v["n"], 3) if k != "LEGIT" else None,
                                   "false_positive_rate": round(v["escalated"] / v["n"], 3) if k == "LEGIT" else None} for k, v in sorted(by_type.items())}
        out["hot_path_latency_ms"] = {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)} for k, v in lat.items()}

        # ── dossier integrity ──
        dossiers = {iid: c["dossier"] for iid, c in full.items() if c.get("dossier")}
        n_d = max(len(dossiers), 1)
        checked = bad = 0
        for d in dossiers.values():
            for e in d["evidence_manifest"]:
                ok = independent_verify(wh, e["record_id"], e["verification_hash"])
                if ok is not None:
                    checked += 1; bad += (not ok)
        notes = [n for d in dossiers.values() for n in d["guardrail_notes"]]
        sla = [d["synthesis_latency_sec"] for d in dossiers.values()]
        out["dossiers"] = {
            "cases_opened": len(full), "dossiers_ready": len(dossiers), "investigation_failures": len(full) - len(dossiers),
            "erp_evidence_items_independently_reverified": checked, "erp_evidence_mismatches": bad,
            "dossiers_with_>=1_verified_evidence_pct": round(100 * sum(bool(d["evidence_manifest"]) for d in dossiers.values()) / n_d, 1),
            "narrative_guardrail_interventions_pct": round(100 * sum(any("Narrative failed" in n for n in d["guardrail_notes"]) for d in dossiers.values()) / n_d, 1),
            "dropped_citation_notes": sum("Dropped" in n for n in notes), "incomplete_investigations": sum("tool failures" in n or "no valid report" in n for n in notes),
            "synthesis_latency_sec": {"p50": round(statistics.median(sla), 2) if sla else None, "p95": round(float(np.percentile(sla, 95)), 2) if sla else None,
                                      "spec_target_lt": 8.0}}
        acts = defaultdict(Counter)
        for iid, d in dossiers.items():
            acts[labels[iid][1]][d["recommended_action"]] += 1
        out["recommended_actions_by_true_type"] = {k: dict(v) for k, v in sorted(acts.items())}

        # ── adversarial: prompt injection planted in vendor e-mail ──
        adv = []
        for iid in adversarial:
            d = dossiers.get(iid)
            if d:
                adv.append({"invoice": iid, "action": d["recommended_action"], "injection_flagged": any("instruction-like" in x for x in d["document_findings"]),
                            "auto_approved": d["recommended_action"] == "AUTO_APPROVE"})
        out["prompt_injection"] = {"cases": len(adv), "flagged": sum(a["injection_flagged"] for a in adv), "auto_approved": sum(a["auto_approved"] for a in adv), "detail": adv}

        # ── simulated analyst → KPIs ──
        for iid, c in full.items():
            if not c.get("dossier") or c["status"] == "DISPOSED":
                continue
            fraud = labels[iid][0] == 1
            rec = c["recommended_action"]
            agree = (rec != "AUTO_APPROVE") if fraud else (rec == "AUTO_APPROVE")
            await h.post(f"{s.CASE_URL}/v1/cases/{c['case_id']}/disposition", json={
                "decision": "CONFIRMED_FRAUD" if fraud else "FALSE_POSITIVE", "analyst_id": "sim.analyst", "rationale": "benchmark oracle (ground-truth label)",
                "accepted_recommendation": bool(agree)})
        out["kpis_simulated_analyst"] = (await h.get(f"{s.CASE_URL}/v1/metrics")).json()
        out["ledger"] = (await h.get(f"{s.AUDIT_URL}/v1/verify")).json()

    # ── ML drift: held-out period vs training baseline ──
    try:
        sc = Tier2Scorer.load()
        te = feats[feats.invoice_id.isin(test_ids)]
        rep = drift_report(sc.b.baseline, {f: te[f].values for f in sc.b.baseline})
        out["drift_test_vs_train"] = {"overall": rep.overall, "max_psi": round(rep.max_psi, 3), "worst_feature": rep.worst_feature,
                                      "features_marginal_or_severe": {k: v["psi"] for k, v in rep.per_feature.items() if v["severity"] != "STABLE"}}
        out["model_version"] = sc.b.version
    except FileNotFoundError:
        pass
    return out


def render_markdown(o: dict) -> str:
    t, dz, hp = o["triage"], o["dossiers"], o["hot_path_latency_ms"]
    L = [f"# Evaluation report — {time.strftime('%Y-%m-%d')}", "",
         f"LLM provider: **{o['meta']['llm_provider']}** · model bundle `{o.get('model_version', 'n/a')}` · sample: {o['meta']['sampled']} held-out invoices "
         f"({o['meta']['fraud']} fraud / {o['meta']['legit']} legit) · **all data synthetic — numbers demonstrate the pipeline, not production accuracy**.", "",
         "## Triage (Tier 1 ∪ Tier 2 escalation vs ground truth)", "",
         f"recall **{t['recall']}** · false-positive rate **{t['false_positive_rate']}** · TP/FP/FN/TN = {t['tp']}/{t['fp']}/{t['fn']}/{t['tn']}", "",
         "| true type | n | escalated | blocked by Tier 1 | recall / FPR |", "|---|---:|---:|---:|---:|"]
    for k, v in o["by_fraud_type"].items():
        L.append(f"| {k} | {v['n']} | {v['escalated']} | {v['tier1_blocked']} | {v['recall'] if v['recall'] is not None else v['false_positive_rate']} |")
    L += ["", f"Cold start (first investigation after boot, excluded from the figures below): {o.get('cold_start_first_investigation_s')}s", "",
          "## Hot-path latency (gateway → scoring → case → audit; wall clock, includes client)", "", "| route | n | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|"]
    for k, v in hp.items():
        L.append(f"| {k} | {v['n']} | {v['p50']} | {v['p95']} | {v['p99']} |")
    L += ["", "Spec target: P99 ≤ 850 ms.", "", "## Tier 3 investigation", "",
          f"dossiers {dz['dossiers_ready']}/{dz['cases_opened']} · synthesis P50 {dz['synthesis_latency_sec']['p50']}s, P95 {dz['synthesis_latency_sec']['p95']}s (target < 8s)",
          f"· ERP evidence re-verified independently: {dz['erp_evidence_items_independently_reverified']} items, **{dz['erp_evidence_mismatches']} mismatches**",
          f"· dossiers with ≥1 verified evidence item: {dz['dossiers_with_>=1_verified_evidence_pct']}% · narrative guardrail interventions: {dz['narrative_guardrail_interventions_pct']}%"
          f" · dropped citations: {dz['dropped_citation_notes']} · incomplete investigations: {dz['incomplete_investigations']}", "",
          "| true type | recommended actions |", "|---|---|"]
    for k, v in o["recommended_actions_by_true_type"].items():
        L.append(f"| {k} | " + ", ".join(f"{a} ×{n}" for a, n in v.items()) + " |")
    pi = o["prompt_injection"]
    L += ["", "## Adversarial: prompt injection planted in a vendor e-mail", "",
          f"cases {pi['cases']} · injection flagged **{pi['flagged']}** · auto-approved **{pi['auto_approved']}**", "",
          "## Agent KPIs (simulated analyst using ground truth)", "", "```json", json.dumps(o["kpis_simulated_analyst"], indent=1), "```", "",
          f"## Audit ledger\n\n{o['ledger']}", ""]
    if "drift_test_vs_train" in o:
        L += ["## ML drift (held-out period vs training baseline)", "", f"```json\n{json.dumps(o['drift_test_vs_train'], indent=1)}\n```", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraud", type=int, default=60)
    ap.add_argument("--legit", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--investigation-timeout", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = asyncio.run(run(a))
    md = render_markdown(res)
    print(md)
    if a.out:
        a.out.write_text(md)
        a.out.with_suffix(".json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
