# Evaluation report — 2026-11-04

LLM provider: **offline** · model bundle `20261104085350-45de14d` · sample: 126 held-out invoices (46 fraud / 80 legit) · **all data synthetic — numbers demonstrate the pipeline, not production accuracy**.

## Triage (Tier 1 ∪ Tier 2 escalation vs ground truth)

recall **0.891** · false-positive rate **0.0** · TP/FP/FN/TN = 41/0/5/80

| true type | n | escalated | blocked by Tier 1 | recall / FPR |
|---|---:|---:|---:|---:|
| BANK_DIVERSION | 14 | 13 | 12 | 0.929 |
| COLLUSION | 5 | 5 | 0 | 1.0 |
| CONTRACT_VIOLATION | 7 | 7 | 0 | 1.0 |
| DUPLICATE | 3 | 2 | 2 | 0.667 |
| LEGIT | 80 | 0 | 0 | 0.0 |
| PHANTOM_DELIVERY | 9 | 9 | 0 | 1.0 |
| SANCTIONED_VENDOR | 1 | 1 | 1 | 1.0 |
| SMURFING | 7 | 4 | 4 | 0.571 |

Cold start (first investigation after boot, excluded from the figures below): 1.91s

## Hot-path latency (gateway → scoring → case → audit; wall clock, includes client)

| route | n | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| APPROVE | 85 | 75.1 | 160.5 | 182.6 |
| ALL | 126 | 79.4 | 165.9 | 235.6 |
| BLOCK | 19 | 83.7 | 168.9 | 226.5 |
| INVESTIGATE | 22 | 89.6 | 215.3 | 244.6 |

Spec target: P99 ≤ 850 ms.

## Tier 3 investigation

dossiers 41/41 · synthesis P50 3.1s, P95 3.91s (target < 8s)
· ERP evidence re-verified independently: 56 items, **0 mismatches**
· dossiers with ≥1 verified evidence item: 97.6% · narrative guardrail interventions: 0.0% · dropped citations: 0 · incomplete investigations: 0

| true type | recommended actions |
|---|---|
| BANK_DIVERSION | HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT ×13 |
| COLLUSION | HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT ×5 |
| CONTRACT_VIOLATION | HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT ×7 |
| DUPLICATE | HOLD_PAYMENT ×2 |
| PHANTOM_DELIVERY | HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT ×9 |
| SANCTIONED_VENDOR | FREEZE_VENDOR ×1 |
| SMURFING | HOLD_PAYMENT ×4 |

## Adversarial: prompt injection planted in a vendor e-mail

cases 6 · injection flagged **6** · auto-approved **0**

## Agent KPIs (simulated analyst using ground truth)

```json
{
 "cases_total": 41,
 "by_status": {
  "INVESTIGATING": 0,
  "PENDING_REVIEW": 0,
  "DISPOSED": 41
 },
 "gross_prevented_loss": 1166025.68,
 "false_positive_operational_cost": 0.0,
 "recommendation_acceptance_rate": 1.0,
 "median_resolution_seconds": 5.7,
 "median_synthesis_latency_sec": 3.1,
 "precision_of_escalations": 1.0
}
```

## Audit ledger

{'valid': True, 'detail': 'ok', 'blocks': 208, 'head': 'da30485ee87d6d2e2de8787e0674fdd3c760f541a4bdbb4281dd3176e21262ac'}

## ML drift (held-out period vs training baseline)

```json
{
 "overall": "SEVERE_DRIFT",
 "max_psi": 0.427,
 "worst_feature": "approver_concentration",
 "features_marginal_or_severe": {
  "bank_age_days": 0.1059,
  "vendor_age_days": 0.1844,
  "extraction_confidence": 0.1891,
  "amount_zscore_vendor": 0.1573,
  "amount_ratio_vendor_mean": 0.2259,
  "rolling_24h_spend_ratio": 0.1864,
  "rolling_7d_spend_ratio": 0.1911,
  "unit_price_ratio_vendor_hist": 0.4127,
  "approver_concentration": 0.427
 }
}
```
