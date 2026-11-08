# Procure-to-Pay Fraud Detection — agentic, microservice MVP

> **🌐 Live use-case page: <https://sadiqgpasha89.github.io/procure2pay-fraud-detection/>** — a visual walkthrough of the problem, the three-tier approach, architecture and results.

Detects and intercepts procure-to-pay fraud (bank-detail diversion, split invoices, shell vendors, collusion, phantom deliveries, contract overbilling, duplicates, sanctioned vendors) **before payment**, and gives investigators an evidence-cited case dossier.

Built on **Google ADK** (agents), **A2A** (agent↔agent), **MCP** (agent↔tools), a **RAG** layer over contracts/policies/e-mails, and a GCP-ready, open-source-by-default stack. Runs end-to-end on a laptop with **synthetic data** and no API key; one switch moves Tier 3 to a local open-weights model or Vertex AI Gemini.

> Write-up: [Part 1 — architecture & fast path](docs/articles/part-1-architecture-and-the-fast-path.md) · [Part 2 — agents, MCP, A2A, evidence](docs/articles/part-2-agents-mcp-a2a-and-evidence.md) · [Part 3 — real LLMs, tests, GCP](docs/articles/part-3-real-llms-testing-and-gcp.md)

> This is v2 of the project. The v1 monolith (templated "agents", hard-coded evidence hashes, a model trained on 10 hand-typed rows) is archived in [`legacy/`](legacy/) and can be deleted.

## Architecture

```
 ERP / payment run
        │  POST /v1/evaluate {invoice_id}
        ▼
 ┌──────────────┐  REST   ┌────────────────────────────┐        ┌───────────────┐
 │   gateway    │────────▶│ scoring-service            │        │ audit-service │  SHA-256 hash chain
 │ idempotent   │         │ Tier 1 rules (µs)          │        │ single writer │  file │ GCS WORM │ BigQuery
 │ decisions    │         │ Tier 2 LightGBM+IForest+   │        └───────▲───────┘
 └──────┬───────┘         │   graph, TreeSHAP          │                │ every decision, dossier, human sign-off
        │ APPROVE / HOLD / BLOCK (<250 ms P99)         │                │
        │ + opens case, then async ▼                                    │
        │            ┌──────────────────────────────────────────┐       │
        │    A2A     │ investigator (ADK custom agent)          │       │
        └───────────▶│  1 fan-out  ──A2A──▶ graph-analyst  ──MCP──▶ mcp-graph      (NetworkX → Neo4j/Spanner)
                     │               ──A2A──▶ doc-auditor   ──MCP──▶ mcp-erp        (SQLite → BigQuery/AlloyDB)
                     │                                       └─MCP──▶ mcp-knowledge  (RAG: TF-IDF ⊕ dense, RRF)
                     │  2 re-verify every cited hash against the owning MCP server
                     │  3 narrator LLM writes prose from VERIFIED findings only
                     │  4 guardrails: citation + grounding check
                     │  5 deterministic policy picks the action → CaseDossier
                     └──────────────────┬───────────────────────┘
                                        ▼
                              case-service (human-in-the-loop, KPIs, console UI)
```

| service | port | role |
|---|---|---|
| `gateway` | 8100 | entry point, idempotency (UUIDv5), routes, dispatches Tier 3 off the hot path |
| `scoring` | 8101 | features → Tier 1 → Tier 2 (+ circuit breaker, SHAP) |
| `case` | 8102 | investigator queue, dispositions, business/agent KPIs, console at `/` |
| `audit` | 8103 | tamper-evident ledger (single writer) |
| `mcp-erp` / `mcp-graph` / `mcp-knowledge` | 8201–8203 | read-only MCP tool servers |
| `investigator` / `graph-analyst` / `doc-auditor` | 8300–8302 | A2A agent servers |

## Quick start

```bash
make setup            # uv venv + deps (Python 3.12)
make bootstrap        # synthetic data → features → models → RAG index  (~25 s)
make up               # 10 services; gateway :8100, investigator console :8102
curl -s localhost:8100/v1/evaluate -H 'content-type: application/json' \
     -d '{"invoice_id":"INV-004065","wait_for_investigation":true}' | jq .dossier.agent_narrative
make benchmark        # replay the held-out period → docs/EVAL_REPORT.md
make test             # 47 tests: unit + full 10-process e2e with auth enabled
```

Docker: `docker compose -f deploy/docker/docker-compose.yml up --build` (add `--profile ollama` for a local LLM).

## Choosing the Tier 3 model (`LLM_PROVIDER`)

| value | what runs | cost | notes |
|---|---|---|---|
| `offline` (default) | deterministic **reference model** that issues real tool calls through ADK | $0 | CI, demos, cold start, and the fallback if an LLM provider is down. Plain narratives, identical safety properties |
| `ollama` | any local open-weights model via LiteLLM (`OLLAMA_MODEL=ollama_chat/gemma4:e4b`) | $0 + GPU | needs `pip install .[ollama]`. See "real-LLM findings" below |
| `vertex` | Gemini on Vertex AI (`GEMINI_MODEL`) | pay-per-token | `pip install .[gcp]`; credentials via ADC / Cloud Run service account |

RAG embeddings are independent: `EMBEDDING_PROVIDER=lexical` (TF-IDF, free) · `ollama` (nomic-embed-text) · `vertex` (text-embedding-005). Hybrid retrieval fuses both by Reciprocal Rank Fusion.

## Design decisions that matter

1. **The LLM never decides money.** The recommended action comes from `agents/policy.py` — a pure function of Tier-1 violations, Tier-2 score and *verified* findings. A failed or empty investigation can never yield `AUTO_APPROVE`.
2. **Evidence is content-addressed.** Every tool result carries `(source, record_id, sha256(record))`. Before a claim reaches the dossier, the investigator asks the originating MCP server to re-derive the hash; unverifiable citations and the claims relying on them are dropped.
3. **Specialists don't see the ML score.** They get only `{invoice_id, vendor_id}` and must corroborate from source systems (independent evidence, no anchoring).
4. **Retrieved text is untrusted.** Instruction-like spans are stripped at the MCP boundary and reported as a finding. The synthetic corpus plants an injection payload in vendor e-mails; the benchmark checks it never steers the outcome.
5. **No train/serve skew.** One `FeatureBuilder` serves training and scoring, always point-in-time (strictly earlier history).
6. **Ground truth is unreachable by agents.** Labels live in a separate store imported only by training/evaluation; a test asserts no tool or agent module can reach it.
7. **Fail closed, stay up.** Circuit breaker adds +0.15 risk when verification APIs fail; audit outage does not stop decisions (it is logged for alerting); tool errors mark the dossier *incomplete*.

## Results (synthetic data — demonstrates the pipeline, not production accuracy)

Temporal hold-out, offline reference model, full report in [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md):

| metric | value |
|---|---|
| Tier 2 AUC-PR (rule-passing population) | 0.89 (base rate 4%) |
| End-to-end recall / legit flag rate (model card, 668 held-out invoices) | 87.8% / 1.75% |
| Benchmark (126 sampled held-out invoices) recall / FPR | 89.1% / 0.0% |
| Hot-path latency, 8 concurrent clients (spec: P99 ≤ 850 ms) | P50 79 ms · P99 236 ms |
| Tier 3 synthesis (spec: < 8 s) | P50 3.1 s · P95 3.9 s |
| ERP evidence independently re-verified | 56 items, 0 mismatches |
| Planted prompt injections flagged / auto-approved | 6 of 6 / 0 |

Weak spots, reported not hidden: split-invoice (smurfing) recall is 57% because the *first* invoice of a split looks normal when it arrives; near-duplicates are caught only inside a 7-day window; the drift monitor reports moderate drift on the held-out period (history warm-up and the enriched benchmark sample).

### Real-LLM findings (gemma4:e4b via Ollama, local, 4 scenarios)

All four produced valid dossiers in 29–69 s with hash-verified evidence and the right action, but only after three fixes this surfaced: thinking mode ate the output-token budget and truncated JSON (`OLLAMA_THINK=false`, 15 s→2.5 s per call); the model skipped tool calls (→ forced first-hop `TOOL_PREFETCH`); and it sometimes dropped a closing brace (→ validated JSON repair). Its *analysis* is looser than the reference model's — e.g. it quoted a contract price cap without comparing it to the invoiced price. Verification proves a cited record exists and is unchanged, **not that the model interpreted it correctly**; analysts see claim and evidence side by side, and the action policy limits blast radius. Treat a larger model (or Gemini) as the production default. A gpt-oss:20b comparison was not run.

## GCP mapping and cost

| capability | open stack (default) | Google Cloud |
|---|---|---|
| agents / LLM | ADK + reference or Ollama model | ADK on Cloud Run (A2A servers) or **Vertex AI Agent Engine** (investigator); Gemini via Vertex AI |
| tools | FastMCP servers | same, on Cloud Run |
| RAG | TF-IDF ⊕ Ollama embeddings, NumPy index | Vertex `text-embedding-005`; pgvector/AlloyDB or Vertex Vector Search behind the same `search()` |
| warehouse / features | SQLite | BigQuery / AlloyDB (replace `Warehouse`); Memorystore for online features |
| models | LightGBM bundle on disk | Vertex AI Training (same `p2p.ml.train` entrypoint) + Model Registry + Endpoints |
| case + idempotency stores | SQLite | Cloud SQL PostgreSQL (`CASE_DB_URL`, `DECISION_DB_URL`) |
| audit | JSONL hash chain | GCS bucket with **locked retention** (create-only objects) + BigQuery mirror |
| secrets / auth | `SERVICE_TOKEN` | Secret Manager; hardening path: internal ingress + OIDC |

Cost levers already in the design: scale-to-zero Cloud Run for all but gateway/scoring; Flash-class Gemini or local open weights; **Tier 3 runs only on the ~7–10% of invoices Tier 1–2 escalate**; specialists get a minimal prompt; lexical RAG needs no embedding spend; one container image for all ten services.

**Verified vs not:** everything in `src/`, the tests, compose-file syntax and the Ollama path were run here. **Not executed** (no GCP project, no `terraform`, no Docker daemon on this machine): `deploy/terraform`, `deploy/cloudbuild.yaml`, `deploy/docker/Dockerfile` build, `deploy/agent_engine/deploy_investigator.py` (API checked against google-cloud-aiplatform 2.7), `GcsSink`/`BigQuerySink`, `VertexEmbedder`, Vertex-hosted Gemini. Review before applying.

## Known gaps (MVP)

- SQLite warehouse is read from a GCS FUSE mount on Cloud Run; production should use BigQuery/AlloyDB via a `Warehouse` implementation (the interface is the seam).
- Service auth is a shared bearer token; production should use internal ingress + OIDC identity tokens.
- OCR/Document-AI ingestion, ERP adapters, Beam/Dataflow streaming and entity resolution from v1 were not carried over (v1 versions were stubs); `Invoice.extraction_confidence` is a feature but no OCR pipeline produces it.
- Synthetic fraud is simpler than real fraud; thresholds τ₁/τ₂ are config, not tuned to a cost curve.
- No UI authentication (`/` console is for local use; put IAP in front).

## Layout

```
src/p2p/
  common/      config, models (Appendix dossier schema), evidence hashing, ops (auth, metrics, circuit breaker), security (PII, injection)
  data/        Warehouse + LabelStore, synthetic world generator
  features/    FeatureBuilder (train = serve), KnowledgeGraph
  rules/       Tier 1 engine
  ml/          train, scorer (+TreeSHAP), fast Isolation Forest, drift (PSI/Wasserstein)
  rag/         chunking, hybrid index, ingest
  mcp_servers/ erp, graph, knowledge
  agents/      investigator, specialists, reference model, policy, narrative guardrails, prompts, A2A apps
  services/    gateway, scoring, case, audit
  llmops/      end-to-end benchmark
  serve.py     stack.py
deploy/        docker, compose, terraform, cloudbuild, agent_engine
docs/          ARCHITECTURE.md, EVAL_REPORT.md, articles/ (3-part write-up)
tests/         unit/ (41) · e2e/ (6, boots 10 processes with auth)
```

Specification: [`procure_to_pay_fraud_detection_use_case.md`](procure_to_pay_fraud_detection_use_case.md).
