import numpy as np
import pytest

from p2p.agents.narrative import check_narrative, template_narrative
from p2p.agents.policy import decide_action
from p2p.agents.schemas import Finding, parse_report
from p2p.common.models import Action, Evidence, RiskClass, Tier1Result, Tier2Result
from p2p.ml.drift import drift_report, make_baseline, severity
from p2p.ml.fast_iforest import FastIForest
from p2p.ml.scorer import Tier2Scorer, classify, composite


def test_fast_iforest_matches_sklearn_exactly(world):
    sc = Tier2Scorer.load(world / "models")
    X = sc.b.scaler.transform(np.random.default_rng(1).normal(size=(64, len(sc.b.feature_order))) * 3)
    assert np.allclose(FastIForest(sc.b.iforest).score_samples(X), sc.b.iforest.score_samples(X), atol=1e-12)


def test_composite_is_noisy_or_and_bounded():
    assert composite(0.2, 0, 0) == pytest.approx(0.2)
    assert composite(0.2, 1, 0) > 0.5 and composite(0, 0, 1) == pytest.approx(0.7)
    assert composite(0.9, 1, 1, penalty=0.5) == 1.0
    assert classify(0.2, 0.35, 0.7) is RiskClass.LOW and classify(0.5, 0.35, 0.7) is RiskClass.MEDIUM and classify(0.9, 0.35, 0.7) is RiskClass.HIGH


def test_model_separates_fraud_from_legit_on_heldout(world, engine, labels, wh):
    ids = wh.invoice_ids()[int(0.8 * len(wh.invoice_ids())):]
    fraud = [engine.score(i).tier2.composite_score for i in ids if labels[i][0]]
    legit = [engine.score(i).tier2.composite_score for i in ids if not labels[i][0]]
    assert np.mean(fraud) > np.mean(legit) + 0.25
    card = Tier2Scorer.model_card(world / "models")
    assert card["data"]["synthetic"] is True and card["metrics"]["tier2_population"]["auc_pr_composite"] > 0.5


def test_shap_contributions_are_positive_sorted_and_sum_to_a_real_logit(engine, labels):
    fraud_ids = [i for i, (f, _) in labels.items() if f][:60]
    top = max((engine.score(i) for i in fraud_ids), key=lambda r: r.tier2.supervised_score)
    c = top.tier2.contributions
    assert c and all(x.shap_value > 0 for x in c) and [x.shap_value for x in c] == sorted((x.shap_value for x in c), reverse=True)


def test_circuit_breaker_penalty_raises_score(engine, by_type):
    iid = by_type["LEGIT"][10]
    base = engine.score(iid).tier2.composite_score
    engine.simulate_outage = True
    try:
        for _ in range(6):
            engine.score(iid)
        degraded = engine.score(iid).tier2
    finally:
        engine.simulate_outage = False
        engine.breaker.reset_timeout = 0.0
        engine.score(iid); engine.breaker.reset_timeout = 30.0
    assert degraded.fallback_penalty == 0.15 and degraded.composite_score >= min(1.0, base + 0.15) - 1e-9


def test_drift_monitor_thresholds():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    q = make_baseline(base)
    same = drift_report({"f": q}, {"f": rng.normal(0, 1, 2000)})
    shifted = drift_report({"f": q}, {"f": rng.normal(1.5, 1, 2000)})
    capped = make_baseline(np.minimum(rng.exponential(400, 5000), 365))  # heavily tied (capped) features must not look drifted when they are not
    assert drift_report({"f": capped}, {"f": np.minimum(rng.exponential(400, 2000), 365)}).overall == "STABLE"
    assert same.overall == "STABLE" and shifted.overall == "SEVERE_DRIFT" and shifted.needs_shadow_retrain
    assert severity(0.05) == "STABLE" and severity(0.15) == "MARGINAL_SHIFT" and severity(0.3) == "SEVERE_DRIFT"


FACTS = {"invoice_id": "INV-000001", "vendor_id": "VEND-0001", "amount": 5000.0, "score": 0.8, "classification": "HIGH_RISK", "rule_violations": []}
FINDINGS = [{"kind": "NO_GOODS_RECEIPT", "severity": "HIGH", "claim": "Invoice INV-000001 has no goods receipt on file for PO PO-000001.",
             "evidence": [{"source": "ERP", "record_id": "PO:PO-000001", "verification_hash": "a" * 64}]}]
MANIFEST = [Evidence(source="ERP", record_id="PO:PO-000001", verification_hash="a" * 64)]


def test_narrative_guardrails_accept_grounded_and_reject_fabrication():
    good = template_narrative(FACTS, FINDINGS)
    assert check_narrative(good, MANIFEST, FACTS, FINDINGS) == []
    assert any("unverified citation" in p for p in check_narrative("x [Evidence: INV:FAKE-1]", MANIFEST, FACTS, FINDINGS))
    assert any("ungrounded identifier" in p for p in check_narrative("Vendor VEND-9999 is linked to EMP-042.", MANIFEST, FACTS, FINDINGS))
    assert any("ungrounded amount" in p for p in check_narrative("The invoice was for $9,999,999.00.", MANIFEST, FACTS, FINDINGS))


def test_report_parser_is_tolerant_but_strict_about_schema():
    ok = '```json\n{"agent":"x","findings":[{"kind":"NO_ANOMALY","claim":"c"}]}\n```'
    assert parse_report(ok, "x").findings[0].kind == "NO_ANOMALY"
    assert parse_report('{"findings":[{"kind":"MADE_UP_KIND","claim":"c"}]}', "x").findings[0].kind == "OTHER_ANOMALY"  # closed vocabulary, coerced
    assert parse_report("no json here", "x") is None


def _f(kind, sev="HIGH"):
    return Finding(kind=kind, severity=sev, claim="c", evidence=[Evidence(source="ERP", record_id="r", verification_hash="h")])


T2 = lambda s: Tier2Result(composite_score=s, classification=RiskClass.HIGH if s >= .7 else RiskClass.MEDIUM)  # noqa: E731


@pytest.mark.parametrize("t1,t2,findings,incomplete,expected", [
    (Tier1Result(is_blocked=True, violations=["SANCTIONS_OFAC_HIT"]), T2(0.0), [], False, Action.FREEZE_VENDOR),
    (Tier1Result(), T2(0.5), [_f("STAFF_VENDOR_LINK")], False, Action.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT),
    (Tier1Result(), T2(0.5), [_f("PROMPT_INJECTION_DETECTED")], False, Action.HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT),
    (Tier1Result(is_blocked=True, violations=["EXACT_DUPLICATE_INVOICE"]), T2(0.4), [], False, Action.HOLD_PAYMENT),
    (Tier1Result(), T2(0.5), [_f("NO_CONTRACT", "MEDIUM")], False, Action.REQUEST_MANUAL_KYC),
    (Tier1Result(), T2(0.5), [], False, Action.AUTO_APPROVE),
    (Tier1Result(), T2(0.5), [], True, Action.REQUEST_MANUAL_KYC),  # failed investigation is never "clean"
])
def test_action_policy_matrix(t1, t2, findings, incomplete, expected):
    assert decide_action(t1, t2, findings, incomplete)[0] is expected


def test_report_parser_keeps_valid_findings_when_model_invents_a_kind():
    from p2p.agents.schemas import parse_report_ex
    txt = ('{"agent":"x","findings":[{"kind":"NO_GOODS_RECEIPT","severity":"HIGH","claim":"ok","evidence":[{"source":"ERP","record_id":"r","verification_hash":"h"}]},'
           '{"kind":"MADE_UP","severity":"HIGH","claim":"odd thing","evidence":[{"bad":1}]}, {"kind":"X"}]}')
    rep, problems = parse_report_ex(txt, "x")
    assert [f.kind for f in rep.findings] == ["NO_GOODS_RECEIPT", "OTHER_ANOMALY"] and rep.findings[1].evidence == []
    assert any("coerced" in p for p in problems) and any("without a claim" in p for p in problems)
    rep, problems = parse_report_ex('{"agent":"x","findings":[{"kind":"NO_ANOMALY","claim":"c","evidence":[', "x")  # truncated → repaired
    assert rep and rep.findings[0].claim == "c" and any("repaired" in p for p in problems)
    rep, problems = parse_report_ex('{"agent":"x","findings":[{"kind":"NO_ANOMALY","claim":"c",,}', "x")  # genuinely malformed → rejected
    assert rep is None and "invalid JSON" in problems[0]


def test_truncated_model_json_is_repaired_but_still_validated():
    from p2p.agents.schemas import parse_report_ex, repair_truncated_json
    cut = '{"agent":"x","findings":[{"kind":"NO_GOODS_RECEIPT","claim":"a \\"quoted\\" word","evidence":[{"source":"ERP","record_id":"r","verification_hash":"h"}]}]'  # final } missing
    rep, problems = parse_report_ex(cut, "x")
    assert rep and rep.findings[0].kind == "NO_GOODS_RECEIPT" and any("repaired" in p for p in problems)
    assert repair_truncated_json('{"a": "unterminated') == '{"a": "unterminated"}'
    assert repair_truncated_json('{"a": 1}') is None and repair_truncated_json('{"a": ]') is None
