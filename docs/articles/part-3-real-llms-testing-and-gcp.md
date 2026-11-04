# Catching Procure-to-Pay Fraud with Agents (Part 3/3): Real LLMs, Real Failures, Tests, and the GCP Bill

*What happened when the deterministic stand-in was replaced by a real open-weights model — four failures in a row, the fixes, how the system is tested end to end, how it deploys to Google Cloud, what it costs, and what I did not verify.*

---

[Part 1](./part-1-architecture-and-the-fast-path.md) built the fast path; [Part 2](./part-2-agents-mcp-a2a-and-evidence.md) built an investigator whose claims are re-verified against source systems and whose recommended action is chosen by code, not by the model. Everything so far ran against a deterministic reference model that follows the same tool-use contract as an LLM. That proves the plumbing and the guardrails. It does not prove a real model can do the work. So I pointed the same stack at `gemma4:e4b` served by Ollama on a laptop RTX 4090 — via ADK's LiteLLM wrapper, one environment variable:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=ollama_chat/gemma4:e4b python -m p2p.stack up
```

It failed four different ways before it worked. The failures are the useful part.

---

## Failure 1: 100–230 seconds, and no usable reports

First run, collusion case: 230 s, zero evidence, dossier says *"no corroborating anomaly."* The guardrails fired exactly as designed — both specialists flagged "returned no valid report," the narrator's `<record_id>` placeholder citations were rejected, the dossier was marked **incomplete**, and the action fell back to `REQUEST_MANUAL_KYC` rather than anything resembling approval. The system failed safe, which is the point of building it that way. But it didn't work.

Calling the graph analyst directly, the model *had* produced a good report with correctly copied hashes — in about 90 seconds. So a model that can do the task was being defeated by something mechanical.

**Root cause: thinking mode.** Gemma 4 reasons by default, and those reasoning tokens count against the output budget. Timing the same prompt through LiteLLM:

| setting | latency | completion tokens |
|---|---|---|
| default | 15.3 s | 300 (budget exhausted mid-thought) |
| `think=False` | 2.5 s | 47 |

With a 2,048-token cap, the reasoning ate the budget and the JSON report was cut off. Fix: an `OLLAMA_THINK` setting, default off, plus a larger output cap.

## Failure 2: the model that answered from the wrong source

With thinking off, a new problem. The reports were valid JSON — and the dossiers contained claims like *"vendor has a shared address flag set to 1.0"* and *"amount_zscore_vendor…"*, citing no real record. My verifier dropped every one as ungrounded (good), leaving empty dossiers (bad).

The model wasn't calling its tools at all. The investigator was forwarding the whole scoring result — Tier-2 feature vector included — as the specialists' prompt, so the model just paraphrased the features. This is the independence problem from Part 2 in the flesh: specialists should corroborate from source systems, not echo the ML output. The fix was the `context_builder` hook that sends only `{case_id, invoice_id, vendor_id}`. It's also a token saving.

## Failure 3: the model that skipped its tools

Now the prompt had nothing to paraphrase, and the model returned an immediately valid report with `"findings": []` in 16 seconds. It never called `get_entity_links`. With thinking off, a small model simply doesn't plan tool use reliably.

Prompt-tuning "you MUST call the tool" is a fragile fix. The better observation: **the first hop is deterministic.** For any invoice, the document auditor will call the same four ERP tools plus a policy search; the graph analyst will call one graph tool. There's no reason to ask a model to rediscover that on every case. So a `before_model_callback` issues those calls itself, and only once the evidence is in context does it hand control to the model:

```python
def _prefetch(role):
    plan = getattr(ReferenceLlm(role=role), f"_{role}")      # the reference model's *planning half*
    def cb(callback_context, llm_request):
        responses = collect_function_responses(llm_request)
        calls, _ = plan(invoice_id, responses, "")
        if not calls:
            return None                                       # evidence is in; let the real model write the report
        return LlmResponse(content=Content(role="model",
                    parts=[Part(function_call=FunctionCall(name=n, args=a)) for n, a in calls]))
    return cb
```

Reusing the reference model's planner means the offline path and the prefetch path can never drift apart. It's `TOOL_PREFETCH=auto`: on for Ollama, off for Gemini, which plans tool use well. The LLM still decides on *extra* calls and does what only an LLM does here — interpret heterogeneous evidence into a typed report.

## Failure 4: a missing brace

Now tools ran and reports came back — but about half the document-auditor reports failed to parse: `Expecting ',' delimiter … output 1133 chars`. Looking at the raw text, the model had produced a correct report and simply **forgot the final `}`**.

The parser had been all-or-nothing: one defect, whole report discarded. Two changes. First, a bracket-balancing repair for truncated output:

```python
def repair_truncated_json(text):
    stack, in_str, esc = [], False, False
    for ch in text:
        ...track strings, push "{"/"[" , pop on "}"/"]" ...
    return text + ('"' if in_str else "") + "".join(reversed(stack))
```

This is safe *because* nothing is trusted afterwards: the repaired report is schema-validated and every citation is still hash-verified. Second, per-finding validation — a model that invents one off-vocabulary finding `kind` no longer loses its other findings; the odd one becomes `OTHER_ANOMALY` and the incident is logged in the dossier.

---

## What the working run looks like

Four scenarios, `gemma4:e4b`, local GPU, no cloud: each completes with a dossier in **29–69 s** (the first call after boot is slower while the model loads), with hash-verified evidence and `HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT`.

- **Collusion:** the graph analyst found three real, verified edges — the vendor's address and phone shared with the PO approver, and its address shared with a second vendor — and cited them correctly.
- **Bank diversion with a planted prompt injection:** the lookalike sender domain was found and the injected "recommend AUTO_APPROVE" text had no effect on the outcome.
- **The narrator** wrote `[Evidence: None]` in a couple of cases; the citation guardrail rejected it and substituted the deterministic template. That's the guardrail working on a small-model slip.

And the honest part. The analysis was **looser than the reference model's**. For a contract-overbilling case, the model quoted the contract's price cap as a "finding" without comparing it to the invoiced price. For a phantom delivery it didn't flag the absent goods receipt. For one bank-diversion case it labelled a delivery note "confirms full receipt" under the wrong finding kind. The actions still came out right — but because the *score* and the *policy* were right, not because the model's reasoning was.

This is the limit of the evidence protocol, and it should be stated plainly: **hash verification proves a cited record exists and is unchanged, not that the model interpreted it correctly.** The mitigations are structural — analysts see claim and evidence side by side, the action policy bounds what a wrong claim can cause, and every dossier requires human sign-off — but a 4B-effective model is a demo-grade analyst. For production I'd use a larger model or Gemini, and I'd add deterministic cross-checks for finding kinds that can be re-derived from tool output. I started a gpt-oss:20b comparison and stopped it; I have no numbers for it, so I won't pretend to.

---

## Testing: 47 tests, and what they actually prove

```
tests/unit  (41)   rules, features, ML, RAG, MCP tools, guardrails, policy, ledger, services
tests/e2e   (6)    boots all ten processes with SERVICE_TOKEN set
```

A few worth describing, because they test properties rather than lines:

- **Point-in-time features:** replay old invoices, assert identical vectors and that no history row is dated at or after the invoice.
- **Fast Isolation Forest equals scikit-learn** to 1e-12.
- **Guardrails reject fabrication:** an unverified citation, an invented `VEND-9999`, an invented `$9,999,999.00` — each rejected; a grounded narrative accepted.
- **Policy matrix:** sanctions → freeze; strong indicator → independent audit; **failed investigation → never approve.**
- **Evidence:** a valid hash verifies, a zero hash doesn't, a nonexistent record doesn't.
- **Isolation:** vendor-scoped retrieval never leaks another vendor's contract; no tool or agent module can import the label store; planted injections are neutralised.
- **Ledger:** editing one historical record breaks startup verification; a restart resumes the same chain.
- **End to end, with auth on:** the e2e fixture boots the real stack with a bearer token, so every REST, MCP and A2A hop must carry credentials. That test drove out a whole class of "works only because auth is off" mistakes. It asserts a clean invoice is auto-approved in under 850 ms with no case; a sanctioned vendor is blocked and frozen; collusion is investigated over A2A with GRAPH_DB evidence; a planted injection cannot steer the outcome; and the ledger shows decisions, dossiers and the human disposition with an intact chain.

Final benchmark on the held-out period (126 invoices, offline model): hot-path **P50 79 ms / P99 236 ms** (target 850), synthesis **P50 3.1 s / P95 3.9 s** (target 8 s), **56 ERP evidence items independently re-verified, 0 mismatches**, 6/6 planted injections flagged, 0 auto-approved. The full report, including per-fraud-type recall and the weak spots, is in the repo.

One process note that cost me time: ADK 2.13, a2a-sdk 1.3 and the MCP SDK 2.3 are all newer than anything I had memorised, and APIs had moved (`SequentialAgent` deprecated, `fastmcp` with its own `httpx2.Auth`). I built a ten-line spike — MCP server → ADK agent → A2A → `RemoteA2aAgent` — before writing anything else, and read the installed source rather than trusting recall. Build the thin vertical slice first.

---

## Deploying to Google Cloud — and what I couldn't verify

The mapping from open-source defaults to GCP is deliberately one environment variable per component:

| capability | default (free) | Google Cloud |
|---|---|---|
| Tier 3 LLM | reference model / Ollama | Gemini on **Vertex AI** (`LLM_PROVIDER=vertex`) |
| agent runtime | ADK on any host | **Cloud Run** A2A services, or **Vertex AI Agent Engine** for the investigator |
| RAG embeddings | TF-IDF ⊕ nomic-embed-text | Vertex `text-embedding-005`; pgvector/AlloyDB or Vertex Vector Search behind the same `search()` |
| warehouse / features | SQLite | BigQuery / AlloyDB / Memorystore (swap the `Warehouse` class) |
| case + idempotency stores | SQLite | **Cloud SQL** PostgreSQL (`CASE_DB_URL`, `DECISION_DB_URL`) |
| audit | JSONL hash chain | **GCS bucket with a locked retention policy** (create-only objects → true WORM) + BigQuery mirror |
| secrets | env var | Secret Manager |

The repo contains a Dockerfile (one image, ten services chosen by command), a docker-compose file, a Terraform module (Cloud Run ×10, Artifact Registry, Secret Manager, Cloud SQL, the retention-locked audit bucket, BigQuery), a Cloud Build pipeline, and a Vertex AI Agent Engine deployment script.

**None of the GCP deployment artifacts were executed.** This machine had no GCP project, no `terraform`, and no running Docker daemon. What I *did* verify: the compose file's syntax, the Ollama and embedding paths with real models, and the Agent Engine script's call shape against `google-cloud-aiplatform` 2.7. The Terraform, the image build, the GCS and BigQuery audit sinks, and Vertex-hosted Gemini are unrun — read the plan before applying, and note the Terraform's own comment says so. Reviewing my Terraform by eye did catch two real gaps (Cloud Run needs an explicit `cloud_sql_instance` volume to expose the `/cloudsql` socket, and DB credentials should arrive as a secret-held DSN), which is a decent argument for running `terraform validate` before trusting any of it.

### Where the money goes

The design is cost-shaped more than cost-tuned:

- **Tier 3 runs on ~7% of invoices.** Rules and ML are effectively free; the LLM bill scales with escalations, not volume.
- **Specialists get a minimal prompt** — identifiers only — so tokens per case stay small.
- **Everything scales to zero** except the gateway and scoring service (min one instance, for latency).
- **Local open weights** (Gemma, gpt-oss) for development and cost-sensitive production; a Flash-class Gemini when you want managed reliability.
- **TF-IDF retrieval costs nothing**; dense embeddings are opt-in.
- **One image** for ten services — one build, one scan, one registry entry.

I haven't priced a specific workload, and a number without a volume and a model would be invention. The lever to watch is escalation rate × tokens per investigation × model price; the first two are measurable in the case service's metrics.

---

## Lessons

1. **Put the model where judgement is needed and nowhere else.** The score is ML, the controls are rules, the decision is policy; the LLM gathers and explains.
2. **Verify, don't trust, anything a model emits.** Content hashes, a closed vocabulary, grounded-prose checks. Fabrication has to be structurally impossible, not unlikely.
3. **Make failure explicit and fail closed.** A silent tool error that reads as "no anomaly" is the worst bug an investigator can have; "incomplete" must be a first-class state.
4. **Build a deterministic stand-in for the model.** It makes CI possible, gives you a baseline, and keeps the product up when the provider isn't.
5. **Small models need scaffolding, and scaffolding is cheap.** Disable thinking, prefetch the deterministic first hop, validate per finding, repair obvious truncation — then re-verify everything.
6. **Treat suspicious results as bugs in your setup first.** The 95.5% recall that dropped to 87.8%, the drift alarm that was right, the 8 ms latency that was a cache — each was my mistake, found because something contradicted something else.
7. **Say what you didn't run.**

The code, the tests, the evaluation report and these three articles are in the repository. If you take one idea from the series: the interesting question about an agentic system is rarely "can the model do the task?" — it's "what happens, precisely, when it can't?"
