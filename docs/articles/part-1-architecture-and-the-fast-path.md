# Catching Procure-to-Pay Fraud with Agents (Part 1/3): Architecture and the Fast Path

*An evidence-cited, agentic fraud platform on Google ADK, A2A and MCP — built so the expensive, slow, fallible part (the LLM) only runs on the ~7% of invoices that need it, and can never move money on its own.*

---

Procure-to-pay (P2P) fraud is boring in the way that makes it expensive. Nobody hacks anything. Someone emails accounts-payable from `harb0rfacilities.com` instead of `harborfacilities.com` and asks them to "update our banking details, urgent, we're travelling." Someone splits a $98,000 invoice into three $49,000 ones to stay under the $50,000 dual-approval limit. A buyer's brother-in-law's "consultancy" shares a phone number with the person who approves its purchase orders. A vendor bills for goods nobody received.

I inherited a repository that claimed to solve this: a "tri-tier" detector with 51 passing tests and a README boasting a 2.47 ms P99. Reading the code told a different story. The "agentic" tier was string templates with **hard-coded evidence hashes** (every dossier cited the same `e3b0c44…` SHA-256 — which is the hash of the empty string). The supervised model was trained on ten hand-typed rows. The "LangGraph state machine" was a Python function. The tests passed because they tested the stubs.

So this series is the refactor: a production-shaped, microservice-based, genuinely agentic MVP using **Google ADK**, **A2A**, **MCP**, a **RAG** layer, GCP services where they earn their keep, and open-source tools wherever they cut cost — end to end on **synthetic data**, with real numbers and the bugs I hit.

- **Part 1 (this one):** the architecture, the synthetic world, the fast path (rules + ML), and the latency and evaluation traps.
- **Part 2:** the agent layer — MCP tools, A2A specialists, RAG, and the evidence protocol that makes hallucinated citations impossible.
- **Part 3:** running it with a real local LLM, what broke, tests, GCP deployment and cost — and what I did *not* verify.

---

## The shape of the system

The core design decision: **put the cheap, fast, deterministic things in front of the expensive, slow, fallible one.**

```
ERP / payment run → gateway ──► scoring-service   Tier 1 rules (microseconds)
                       │                          Tier 2 LightGBM + Isolation Forest + graph
                       │   APPROVE / HOLD / BLOCK in ~80 ms (P50)
                       │
                       └─ async ─► investigator ──A2A──► graph-analyst ──MCP──► mcp-graph
                                        │        ──A2A──► doc-auditor   ──MCP──► mcp-erp, mcp-knowledge (RAG)
                                        ▼
                                  case-service  ←→  audit-service (hash-chained ledger)
```

Ten processes, one container image (the service is chosen by the command):

| service | job |
|---|---|
| `gateway` | single entry point; idempotent (UUIDv5 per invoice); dispatches Tier 3 *off* the hot path |
| `scoring` | features → Tier 1 → Tier 2; stateless, CPU-bound, never calls an LLM |
| `investigator`, `graph-analyst`, `doc-auditor` | ADK agents, each its own A2A server |
| `mcp-erp`, `mcp-graph`, `mcp-knowledge` | read-only MCP tool servers |
| `case` | human-in-the-loop queue, dispositions, KPIs, a small console |
| `audit` | the only writer of a SHA-256 hash-chained forensic ledger |

Why split it this way? Because the pieces have opposite scaling profiles. Scoring is CPU-bound and latency-critical: keep warm instances, scale on CPU, never put a network call to a model in its path. The agents are I/O- and LLM-bound, slow and bursty: scale to zero, and keep them as separate services so each can use a different model, prompt, and tool allow-list. The audit ledger has exactly one writer because a hash chain needs a total order.

The gateway's contract is the important one. It returns a decision **before** any LLM runs:

| route | immediate action | then |
|---|---|---|
| Tier 2 says low risk | `AUTO_APPROVE` | nothing |
| Tier 2 medium/high | `HOLD_PAYMENT` | open case, investigate asynchronously |
| Tier 1 hard rule | `BLOCK_TRANSACTION` | open case, investigate asynchronously |

The payment run never waits on a model. The investigation arrives later and is attached to the case for a human.

---

## A synthetic world you can trust enough to measure

No real data, so the first deliverable is a generator (`p2p/data/synthetic.py`) that builds a deterministic world from one seed: 251 vendors, 32 employees, a year of PO → goods receipt → invoice flow (4,349 invoices), bank-change logs, and 852 unstructured documents — supply contracts with price-cap clauses, policies, vendor e-mails, delivery notes.

Into that, eight fraud scenarios are injected (5.3% of invoices):

`BANK_DIVERSION` (lookalike-domain e-mail, change inside *or just outside* the 72-hour freeze) · `SMURFING` · `SHELL_VENDOR` (young vendor, shared bank account/address, round amounts, no goods receipt, buyer approves own PO) · `COLLUSION` (vendor operates from the approver's address and phone, invoices inflated 5.5–9.5% — just under the 10% hard ceiling) · `PHANTOM_DELIVERY` · `CONTRACT_VIOLATION` (billed above the contractual price cap) · `DUPLICATE` (exact and near) · `SANCTIONED_VENDOR`.

Two decisions here paid off later:

**Legit noise is injected too** — young-but-real vendors, legitimate bank changes (a couple of which land inside the 72-hour window, producing genuine Tier-1 false positives), large valid invoices, off-hours work. Without it, every metric is a tautology.

**Labels live in a separate table, behind a separate class.** `Warehouse` serves data; `LabelStore` serves ground truth, and only training and evaluation import it. Later I wrote a test that asserts, by reading source, that no MCP server or agent module can reach it. An agent that can query the answer key isn't being evaluated.

### A flaw in my own world

My first run looked great: 95.5% end-to-end recall. Then the drift monitor reported *severe* drift between train and test, and I assumed the monitor was wrong. It wasn't. In the generator, "young" vendors — including **every shell company** — were registered 20–90 days before the end date, so they only existed in the test window. The model never trained on a shell vendor; it caught them through graph features alone, and my train/test comparison was confounded. After spreading registrations across the timeline, recall dropped to a more honest 87.8%, and the drift fell from PSI 1.2 to 0.4. **If your monitoring contradicts your results, believe the monitoring first.**

(I also had to fix the drift monitor itself: my first PSI re-derived the baseline proportions from 101 quantile knots, which is ill-defined for heavily tied, capped features. The model bundle now stores the actual bin edges and expected proportions measured on the training sample.)

---

## One feature definition for training and serving

The most common silent killer in fraud ML is train/serve skew: features computed one way in a notebook, another way in production. Here there is exactly one `FeatureBuilder`, and it is *point-in-time by construction*:

```python
# p2p/features/builder.py (abridged)
def build(self, inv: Invoice) -> TxContext:
    t = inv.invoice_date                      # "now" is the invoice's own timestamp
    hist365 = wh.vendor_invoices(vendor.vendor_id, t, since=t - timedelta(days=365),
                                 exclude=inv.invoice_id)   # strictly earlier than t
    hist90  = [h for h in hist365 if h.invoice_date >= t - timedelta(days=90)]
    ...
    r24 = sum(h.total_amount for h in w24) + inv.total_amount
    f["smurfing_pattern"] = float(inv.total_amount < lim and r24 >= lim and len(w24) > 0)
```

Every history query takes an explicit `before` timestamp. Training replays all 4,349 invoices through this class in time order; the online scorer calls the same method. A unit test replays old invoices and asserts identical vectors, and that no history row is dated at or after the invoice.

The 30 features span document signals (PO variance, missing goods receipt, bank-change age, threshold proximity to the $50k limit, segregation-of-duties violations), behaviour (90-day z-score, rolling 24-hour spend per vendor and cost centre, near-duplicates, unit price versus the vendor's own history), and graph links (does this vendor share a bank account, address or phone with another vendor — or with the staff who created, approved, keyed or received *this* transaction?).

On the graph: edges form only on **exact normalised matches of a registered attribute**. Two vendors sharing only a director's name get no edge — that is the "supplier name collision" edge case, and a test pins it.

---

## Tier 1: rules that never go down

Five rules, each a small class: exact duplicate, bank change inside 72 hours, sanctions hit, PO ceiling exceeded, split-invoice pattern. They run on the already-assembled features in about 5 microseconds (P50; 12 µs P99), and a rule that throws is logged and skipped — a broken rule must never take down the payment gate.

---

## Tier 2: three signals, one score

- **LightGBM** (`scale_pos_weight` for imbalance, Platt-calibrated on a validation slice) — the supervised signal.
- **Isolation Forest** — trained without labels; only the top-5% tail counts. This is the cold-start and novel-vector hedge.
- **Graph-link score** from the link flags.

Combined as a **noisy-OR**, so independent evidence can only add risk and nothing can exceed 1:

```python
def composite(sup, anom, graph, penalty=0.0):
    s = 1.0 - (1.0 - sup) * (1.0 - 0.5 * anom) * (1.0 - 0.7 * graph)
    return min(1.0, max(0.0, s + penalty))
```

Explanations are exact TreeSHAP values straight from LightGBM (`predict(..., pred_contrib=True)`) — no extra dependency, and every escalated case carries its top drivers.

Two methodology choices keep the numbers honest. The split is **temporal** (65/15/20, never shuffled) after a 90-day history warm-up, so the model is always judged on the future. And Tier 2 is trained and evaluated **only on invoices that pass Tier 1** — that is the population it will actually see; scoring it on frauds the rules already caught inflates everything.

Held-out result (668 invoices, 41 fraud): Tier 2 AUC-PR 0.89 against a 4% base rate; end-to-end recall 87.8% at a 1.75% flag rate on legitimate invoices. The weak spot is real and structural: **split-invoice recall is 57%**, because the first invoice of a split looks perfectly normal when it arrives. Only the second one reveals the pattern.

---

## The latency trap

The spec asks for P99 ≤ 850 ms. My first end-to-end benchmark, eight concurrent clients, said P50 355 ms and P99 **2.2 s**. Profiling scoring on its own showed 33 ms per invoice, and 53 ms of that — nearly all of it — was `IsolationForest.score_samples` on a **single row**. scikit-learn pays roughly 50 µs of joblib/Python overhead *per tree per call*; with 300 trees, that's the whole budget.

The fix is to precompute, for every node of every tree, `depth(node) + c(n_samples(node))` once, so scoring a row is one `tree_.apply` per tree plus a lookup:

```python
class FastIForest:
    def __init__(self, forest):
        for tree, feats in zip(forest.estimators_, forest.estimators_features_):
            t = tree.tree_
            depth = ...                                   # walk the tree once
            self.tables.append((t, depth + _c(t.n_node_samples), ...))
    def score_samples(self, X):
        total = np.zeros(len(X))
        for t, table, _ in self.tables:
            total += table[t.apply(X32)]
        return -(2.0 ** (-total / self.norm))
```

It matches scikit-learn **exactly** (max absolute difference 0.0 — and a test keeps it that way), at ~1.2 ms per row instead of 53. Pinning LightGBM to one thread (OpenMP fan-out costs more than a one-row prediction and thrashes under concurrency) and reusing a long-lived read-only SQLite connection per thread took a scoring call from 33 ms to about 4 ms (Tier 2 itself is 3.4 ms of that).

Two more tail-latency lessons, both about *startup*, not steady state. With everything warm, the gateway's P99 under 8-way concurrency was already ~130 ms — yet the benchmark showed 1.5 s P99. The culprit was the gateway's own first A2A call: resolving the remote agent card and opening the transport blocked its event loop right as traffic started. Services now warm their dependencies in their lifespan hooks (the scoring service scores a few invoices; the gateway pings the investigator), and the stack starts in dependency waves so each can warm its upstreams. Final hot path: **P50 79 ms, P99 236 ms** with the whole stack — gateway, scoring, case creation and a hash-chained audit write — in the loop.

(And a benchmarking trap: one of my reruns showed 8 ms P50. That was idempotency doing its job — I'd forgotten to clear the decision store, so every request was a cache hit. Suspiciously good numbers deserve the same scrutiny as bad ones.)

---

That's the fast path: a decision in well under a second, with an explanation, for every invoice. But the interesting part — and the part most likely to go wrong — is what happens to the ~7% that get escalated.

**Part 2** builds the investigator: MCP tool servers, A2A specialists, RAG over contracts and e-mails, and an evidence protocol that makes a fabricated citation structurally impossible.
