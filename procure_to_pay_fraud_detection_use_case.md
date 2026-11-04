# Enterprise Specification: Procure-to-Pay (P2P) Fraud Detection & Prevention

**Document Version:** 1.0.0  
**Target Audience:** Enterprise Architects, ML Engineers, Security Operations, AP/Internal Audit Leadership  
**Classification:** Internal Technical Architecture & Functional Specification  

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Canonical Procure-to-Pay Lifecycle & Threat Surfaces](#2-canonical-procure-to-pay-lifecycle--threat-surfaces)
3. [Tri-Tier Hybrid Detection Framework](#3-tri-tier-hybrid-detection-framework)
4. [System Architecture & Cloud Engineering](#4-system-architecture--cloud-engineering)
5. [Technical Implementation Details](#5-technical-implementation-details)
6. [Scalability, Resilience & Cloud Blueprints](#6-scalability-resilience--cloud-blueprints)
7. [Comprehensive Metric & Measurement Framework](#7-comprehensive-metric--measurement-framework)
8. [Dual Drift Detection (ML & Agentic AI)](#8-dual-drift-detection-ml--agentic-ai)
9. [Edge Case Analysis & Mitigation Matrix](#9-edge-case-analysis--mitigation-matrix)
10. [Implementation Governance & Rollout Roadmap](#10-implementation-governance--rollout-roadmap)

---

## 1. Executive Summary & Problem Statement

### 1.1 Overview
Procure-to-Pay (P2P) fraud occurs when bad actors exploit organizational procurement, vendor onboarding, and accounts payable (AP) processes to siphon funds, forge invoices, manipulate master vendor records, or bypass compliance mandates. Threats span internal collusion, shell companies, invoice alterations, duplicate billings, kickbacks, and unauthorized routing alterations.

The primary objective of this architecture is to **detect, intercept, and mitigate P2P fraud in near real-time** prior to payment disbursement while strictly bounding false positives and preserving unhindered business operations.

### 1.2 Scope of Operations
The solution protects the full lifecycle of expenditure:
*   Requisitioning & Sourcing
*   Purchase Order (PO) Issuance
*   Goods/Service Receipt (GR)
*   Invoice Intake & Ingestion
*   2-Way / 3-Way Reconciliation Matching
*   Multi-Tier Approval Chains
*   Payment Execution & General Ledger (GL) Posting

### 1.3 Threat Vectors

| Threat Vector | Description | Primary Impact Surface |
| :--- | :--- | :--- |
| **Shell / Fictitious Suppliers** | Onboarding non-existent entities controlled by internal/external actors. | Supplier Master Data, Requisitions |
| **Duplicate / Altered Invoices** | Submitting repeated billing variants or altered totals/banking information. | Invoice Ingestion, Matching Engine |
| **PO Mismatch & False GR** | Fabricating delivery receipts or inflating line items beyond PO tolerances. | Goods Receipt, 3-Way Match |
| **Internal Collusion** | Approvers colluding with vendors to authorize fraudulent or unverified payments. | Approval Chains, Routing Gates |
| **Payment Diversion** | Intercepting vendor profiles to modify settlement accounts (Account Takeover / BEC). | Bank Master Records, AP Release |
| **Policy Circumvention** | Splitting requisitions below approval limits or making uncontracted rogue purchases. | Requisitions, PO Governance |

### 1.4 Technical & Operational Constraints
*   **Throughput & Latency:** Sub-second response requirements during high-concurrency payment release runs.
*   **Heterogeneous ERP Landscape:** Ingestion across decentralized ERPs (SAP S/4HANA, Oracle NetSuite, Workday, Coupa).
*   **Regulatory Compliance:** Full auditability under Sarbanes-Oxley (SOX §404), GDPR, and local statutory tax codes.
*   **Business Continuity:** False positive rates must remain within manageable thresholds to prevent operational bottlenecks.

---

## 2. Canonical Procure-to-Pay Lifecycle & Threat Surfaces

Every stage within the canonical flow exposes unique vulnerabilities and requires distinct detection gates:

```
[ Requisition ] ────────> [ Sourcing / Vendor Onboarding ] ───> [ PO Creation ]
       │                                     │                         │
       ▼                                     ▼                         ▼
Split-order evasion               Shell company onboarding         Off-contract POs
       │                                     │                         │
       └─────────────────────────────────────┼─────────────────────────┘
                                             │
                                             ▼
[ Goods/Service Receipt ] ───> [ Invoice Intake (OCR/EDI) ] ──> [ 2/3-Way Matching ]
       │                                     │                         │
       ▼                                     ▼                         ▼
Phantom deliveries                Tampered bank details / amounts   Mismatch tolerances
       │                                     │                         │
       └─────────────────────────────────────┼─────────────────────────┘
                                             │
                                             ▼
[ Approval Workflow ] ───────> [ Payment Execution ] ─────────> [ Reconciliation & GL ]
       │                                     │                         │
       ▼                                     ▼                         ▼
Collusive approvals               Account diversion / Smurfing      Ghost journals
```

### Stage Breakdown
1. **Requisition:** User initiates goods/service requests. *Risk: Circumvention of spend limits via split orders.*
2. **Sourcing / Supplier Selection:** Onboarding, diligence, and contract filing. *Risk: Inadequate KYC, phantom suppliers.*
3. **PO Creation:** Formal purchase orders generated. *Risk: Fabricated POs, unbudgeted commitments.*
4. **Order Fulfillment / Goods Receipt:** Material receipt confirmation. *Risk: Ghost receipts without physical delivery.*
5. **Invoice Ingestion:** Multi-channel intake (EDI, portal, scanned PDF, email). *Risk: OCR tampering, spoofed domains.*
6. **Invoice Validation / Matching:** Algorithmic verification against PO and GR. *Risk: Exploitation of micro-variance thresholds.*
7. **Approval Workflow:** Delegation-of-authority reviews. *Risk: Bypassed approvals, weekend batch rubber-stamping.*
8. **Payment Execution:** File generation (ACH, Wire, SEPA, SWIFT). *Risk: Mid-cycle bank changes, intercepted routing.*
9. **Reconciliation & Post-Audit:** General ledger balancing and variance reconciliation. *Risk: Concealed manual balance adjustments.*

---

## 3. Tri-Tier Hybrid Detection Framework

To achieve high recall without crippling operations, the platform integrates three complementary computation tiers:

```
Incoming P2P Event (PO, Invoice, Bank Update, Payment)
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ Tier 1: Deterministic Rule Engine (RBS) │
   └────────────────────┬────────────────────┘
                        │
       ┌────────────────┴────────────────┐
   Hard Violation?                  Clean / Ambiguous
       │                                 │
       ▼                                 ▼
[ Immediate Block ]          ┌───────────────────────────────────┐
[ High-Sev Ticket ]          │ Tier 2: Probabilistic ML Scoring  │
                             └─────────────────┬─────────────────┘
                                               │
                       ┌───────────────────────┼───────────────────────┐
                  Low Risk                Medium Risk              High Risk
                  Score < τ₁              τ₁ ≤ Score < τ₂          Score ≥ τ₂
                       │                       │                       │
                       ▼                       └───────────┬───────────┘
               [ Auto-Approve ]                            │
                                                           ▼
                                        ┌─────────────────────────────────────┐
                                        │ Tier 3: Agentic AI Orchestration    │
                                        │ (Graph RAG, Evidence Synthesis)    │
                                        └──────────────────┬──────────────────┘
                                                           │
                                                           ▼
                                        ┌─────────────────────────────────────┐
                                        │ Human-in-the-Loop Case Management   │
                                        │ (Triage, Action Sign-off, Audit)    │
                                        └─────────────────────────────────────┘
```

### 3.1 Tier 1: Deterministic Rule-Based System (RBS)
*   **Characteristics:** Sub-millisecond execution, zero variance, explainable deterministic logic.
*   **Core Responsibilities:** Execute binary policy checks and statutory boundaries.
*   **Trigger Examples:** Exact duplicate invoice identifiers, sanction list hits (OFAC, PEP), unapproved bank changes applied within $\le 72$ hours of payment run, invoices exceeding PO ceilings beyond hard limit.
*   **Enforcement:** Generates immediate transaction blocks or high-priority holds.

### 3.2 Tier 2: Probabilistic Machine Learning (ML)
*   **Characteristics:** Multidimensional scoring, pattern recognition, anomaly isolation.
*   **Core Responsibilities:** Detect behavioral variances against baseline profiles.
*   **Model Suite:**
    *   *Supervised Classifiers:* Gradient Boosted Decision Trees (XGBoost, LightGBM) trained on confirmed historic fraud vectors.
    *   *Unsupervised Ensembles:* Isolation Forests and Autoencoders detecting multidimensional anomalies in invoices lacking explicit fraud labels.
    *   *Graph Neural Networks (GNN) & Node Embeddings:* GraphSAGE / DeepWalk to detect circular payment loops and entity collisions.
*   **Output:** Continuous risk score $S \in [0.0, 1.0]$, feature contribution vectors via SHAP values, and classification tier (Low, Medium, High).

### 3.3 Tier 3: Agentic AI Orchestrator & LLM Specialist Network
*   **Characteristics:** Autonomous, stateful investigation driven by Large Language Models orchestrated via deterministic control loops.
*   **Core Responsibilities:**
    *   Corroborate medium/high ML anomalies across decentralized documentation.
    *   Query enterprise Knowledge Graphs for implicit relationships (e.g., shared phone numbers or corporate addresses between an AP clerk and a supplier director).
    *   Analyze unstructured attachments (emails, vendor contract terms, delivery dockets).
    *   Compile structured Case Synthesis Dossiers containing complete evidence trails, risk hypotheses, and actionable remediation choices.
*   **Guardrails:** The Agentic layer cannot unilaterally delete suppliers or divert funds. All high-impact interventions mandate Human-in-the-Loop (HITL) authorization.

---

## 4. System Architecture & Cloud Engineering

### 4.1 Logical Component Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & NORMALIZATION LAYER                                                             │
│ ERP Gateways (SAP/Oracle) | EDI Ingest | Document OCR/Parsing | Bank Webhooks | Email Parsers  │
└──────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                               │
┌──────────────────────────────────────────────▼─────────────────────────────────────────────────┐
│ 2. STREAM FABRIC & ENRICHMENT BUS                                                              │
│ Kafka / GCP Pub/Sub / AWS Kinesis | Entity Resolution | Sanctions & Registry Enrichment        │
└───────────────────────┬────────────────────────────────────────────────┬───────────────────────┘
                        │                                                │
┌───────────────────────▼────────────────────────┐ ┌─────────────────────▼───────────────────────┐
│ 3. DETERMINISTIC ENGINE                        │ │ 4. FEATURE STORE (Online / Offline)          │
│ In-Memory Policy Checks & Validation Engine    │ │ Low-Latency Key-Value (Redis) + Cold Storage │
└───────────────────────┬────────────────────────┘ └─────────────────────┬───────────────────────┘
                        │                                                │
                        └───────────────────────┬────────────────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────────────────────────────────────────┐
│ 5. ADVANCED ANALYTICS ENGINE                                                                   │
│ ML Model Serving Cluster (Vertex/SageMaker) ── Graph Database (Neo4j/Neptune/AlloyDB)         │
└───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────────────────────────────────────────┐
│ 6. COGNITIVE ORCHESTRATION & AGENT FABRIC                                                      │
│ Agentic Orchestrator (LangGraph/Temporal) ── LLM Gateway ── Enterprise RAG Pipelines           │
└───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────────────────────────────────────────┐
│ 7. GOVERNANCE, AUDIT & INTERACTION                                                             │
│ Investigator UI / Triage Portal | SIEM & Case Sync | Writeback Event Bus | Immutable Audit Log │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cloud Reference Deployments

```
Component                 Google Cloud Platform (GCP)           Amazon Web Services (AWS)
─────────────────────────────────────────────────────────────────────────────────────────────
Stream Ingestion          Pub/Sub                               Kinesis Data Streams / MSK
Stream Compute            Dataflow (Apache Beam)                AWS Glue Streaming / Flink
Document Extraction       Document AI                           AWS Textract
Feature Store (Online)    Memorystore (Redis) / Bigtable        ElastiCache (Redis) / DynamoDB
Feature Store (Offline)   BigQuery                              Amazon Redshift / S3 Lakehouse
ML Training & Serving     Vertex AI Pipelines & Endpoints       Amazon SageMaker Endpoints
Graph Engine              AlloyDB / Neo4j on GKE                Amazon Neptune
Agentic Orchestration     Cloud Run + LangGraph Engine          AWS Fargate / Temporal Core
LLM Foundation Tier       Vertex AI (Gemini Flash/Pro)          Amazon Bedrock (Claude / Titan)
Relational State Storage  Cloud SQL (PostgreSQL)                Amazon RDS Aurora (PostgreSQL)
Object Storage            Cloud Storage (GCS)                   Amazon S3
Secrets & Keys            Cloud KMS & Secret Manager            AWS KMS & Secrets Manager
Audit Trail               Cloud Audit Logs + BigQuery Immutable AWS CloudTrail + S3 Object Lock
```

---

## 5. Technical Implementation Details

### 5.1 Data Ingestion, Normalization & Entity Resolution
*   **Document Processing Pipeline:** Unstructured invoices (PDF/TIFF) traverse an asynchronous OCR and layout-analysis pipeline. Extracted fields include `vendor_tax_id`, `bank_iban_routing`, `line_items`, `net_amount`, and `invoice_date`, each tagged with an extraction confidence score $C_e \in [0.0, 1.0]$. If $C_e < 0.85$, the document routes to an automated data validation queue.
*   **Entity Deduplication & Matching:** Vendor identities undergo multi-field fuzzy matching utilizing Levenshtein distance and Jaro-Winkler algorithms over normalized business names, addresses, and tax identifiers.
*   **External Data Enrichment:** Transactions are supplemented in-stream with external data:
    *   Bank validation APIs (routing correctness, active account flags).
    *   Corporate registry lookups (business registration age, company officer lists).
    *   OFAC/Sanctions list continuous screening.

### 5.2 Feature Engineering Architecture
Features compute across three distinct time domains and are registered inside a centralized Feature Store:

```
Category         Feature Signature                             Domain / Latency
───────────────────────────────────────────────────────────────────────────────────────
Real-Time        abs(invoice_amount - po_amount) / po_amount   Stream (< 20ms)
Real-Time        is_new_bank_account_72h                       Cache hit (< 5ms)
Behavioral       vendor_zscore_amount_last_90d                 Batch aggregation (Redis)
Behavioral       invoices_submitted_outside_business_hours_30d Batch aggregation (Redis)
Graph/Network    degree_centrality_vendor_to_employees         Graph computation (Neptune)
Graph/Network    shared_bank_account_cross_vendor_flag         Graph edge inspection
```

### 5.3 Machine Learning Pipeline & Explainability
*   **Supervised Anomaly Scorer:** An ensemble of LightGBM classifiers trained on balanced historical data (using SMOTE for minority fraud cases) outputs an operational fraud probability score:
    $$P(\text{Fraud} \mid \mathbf{x}) = \sigma\left(\sum_{k} f_k(\mathbf{x})\right)$$
*   **Unsupervised Cold-Start Subsystem:** In scenarios with sparse historical labels, an Isolation Forest calculates isolation depths alongside a deep autoencoder that flags reconstruction errors:
    $$\mathcal{L}_{\text{AE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$
*   **Local Explainability Layer:** Every scored inference runs through a TreeSHAP explainer to isolate top contributing risk variables (e.g., *+0.42 to risk score driven by uncharacteristic invoice amount spike relative to 90-day moving average*).

### 5.4 Agentic AI Orchestration Engine
When a transaction scores in the medium-to-high risk tier ($\tau_1 \le S < 1.0$), the **Agentic Orchestrator** initiates a deterministic state machine:

```
                  ┌───────────────────────────────┐
                  │ 1. Initialize State Machine   │
                  └──────────────┬────────────────┘
                                 │
                  ┌──────────────▼────────────────┐
                  │ 2. Query Context Engine       │
                  │ - Neo4j Link Analysis         │
                  │ - ERP 12-Month Invoicing Hist │
                  │ - Communication / Email Logs  │
                  └──────────────┬────────────────┘
                                 │
                  ┌──────────────▼────────────────┐
                  │ 3. Execute Investigative LLM  │
                  │ - Check Shared Entity Linkages│
                  │ - Validate Delivery Evidence  │
                  │ - Cross-Check Contract Terms  │
                  └──────────────┬────────────────┘
                                 │
                  ┌──────────────▼────────────────┐
                  │ 4. Synthesize Investigation   │
                  │ - Assemble Structured JSON    │
                  │ - Attach Deterministic Citations
                  │ - Calculate Recommendation    │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
              [ Dispatch to Human Investigator UI ]
```

#### Guardrails & Execution Constraints:
1.  **Strict Prompt Boundaries:** The LLM cannot emit raw SQL or call state-changing ERP write APIs directly. It interacts only through read-only tools and verified microservice endpoints.
2.  **Citation Enforcers:** Every claim in the generated narrative must map to a verifiable parameter in the retrieved payload (e.g., `[Evidence: Neo4j Node Edge #4412]`). Unsubstantiated assertions are rejected by an output validation layer.
3.  **Action Isolation:** Recommended remediation choices (e.g., `HOLD_PAYMENT`, `FREEZE_VENDOR`, `REQUEST_MANUAL_KYC`) are emitted as strongly-typed schema elements, not arbitrary strings.

---

## 6. Scalability, Resilience & Cloud Blueprints

```
High-Volume Ingestion Layer
─────────────────────────────────────────────────────────────────────────────────────────────
Throughput Target: 5,000+ operations/sec during peak month-end settlement windows.
Partitioning: Distributed partitioning by `Vendor_Hash_ID` and `Entity_Code` on message queues.

Microservice Elasticity
─────────────────────────────────────────────────────────────────────────────────────────────
Execution Models: Containerized stateless runtimes on AWS EKS or GCP GKE.
Autoscaling: Driven by CPU/Memory thresholds and queue lag metrics (KEDA).

Fault Tolerance & Resilience
─────────────────────────────────────────────────────────────────────────────────────────────
Circuit Breakers: Hystrix/Resilience4j boundaries around all third-party lookup APIs.
Graceful Degradation: If vendor verification times out, cache values are used and the transaction
                      is assigned a fallback penalty score ($S_{\text{penalty}} = +0.15$).
Storage Idempotency: All payment events run with immutable idempotency keys ($UUIDv5$) to
                     prevent accidental duplicate evaluations.
Disaster Recovery: Active-Passive cross-region failover with RPO < 1 min, RTO < 15 mins.
```

---

## 7. Comprehensive Metric & Measurement Framework

To measure technical and business efficacy, the solution tracks metrics across four functional dimensions:

### 7.1 Business Impact Metrics
*   **Gross Prevented Loss:** Total monetary volume of fraudulent disbursement attempts halted prior to settlement:
    $$\sum \text{Value}(\text{True Positive Blocks})$$
*   **False Positive Operational Drag:** Cost of manual investigator overhead wasted reviewing false positives:
    $$\text{FP Cost} = N_{\text{FP}} \times (\text{Average Investigator Hours} \times \text{Blended Hourly Rate})$$
*   **Fraud Resolution Velocity:** Median duration from anomaly trigger to formal disposition (target: $< 2$ hours for high risk).
*   **Under-Billing / Duplicate Leakage Rate:** Percent of duplicate billing missed, calculated post-settlement via internal audits.

### 7.2 Machine Learning Performance Metrics
*   **Area Under Precision-Recall Curve (AUC-PR):** Prioritized over AUC-ROC due to the severe class imbalance inherent to fraud data.
*   **Precision @ Threshold $k$ ($P@k$):** Assesses the accuracy of alerts in the highest confidence deciles.
*   **Brier Calibration Score:** Evaluates whether model probabilities reflect empirical risk:
    $$\text{Brier} = \frac{1}{N}\sum_{t=1}^N (f_t - o_t)^2$$
*   **SHAP Drift Index:** Feature attribution variance across moving 30-day windows.

### 7.3 Agentic & LLM System Metrics
*   **Recommendation Acceptance Rate (RAR):** Proportion of agent triage recommendations accepted by human analysts without manual plan divergence:
    $$\text{RAR} = \frac{\text{Accepted Recommendations}}{\text{Total Agent Dispatches}}$$
*   **Factual Hallucination Rate (FHR):** Cases where post-audit reveals agent-asserted evidence contradicts source systems (target: $< 0.1\%$).
*   **Context Window Efficiency:** Average input/output token utilization per investigated case dossier.
*   **End-to-End Synthesis Latency:** Time consumed to execute graph queries, retrieve docs, and draft the triage dossier (target: $< 8$ seconds).

### 7.4 System Engineering Metrics
*   **End-to-End Decision Latency:** $P_{99} \le 850\text{ms}$ for real-time rule and ML scoring.
*   **Ingestion Pipeline Lag:** Measured queue lag across streaming topics.
*   **Availability SLA:** Target $99.99\%$ uptime for transaction scoring gateways.

---

## 8. Dual Drift Detection (ML & Agentic AI)

```
                       Continuous Monitoring Pipeline
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   [ ML System Drift ]                                 [ Agentic System Drift ]
         │                                                       │
  • Population Stability Index (PSI)                     • Acceptance Rate Drift
  • Wasserstein / Earth Mover's Distance                 • Output Assertion Hallucination Rate
  • Latency & Inference Degradation                      • Token & Cost Inflation
  • Delayed Ground-Truth Evaluation                      • Prompt Injection / Format Deviation
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     ▼
                    Automated Alerting & Rollback Bus
```

### 8.1 Machine Learning Drift Management
1.  **Feature / Covariate Shift:** Compute the Population Stability Index (PSI) daily across all scoring features:
    $$\text{PSI} = \sum \left( (Actual\% - Expected\%) \times \ln\left(\frac{Actual\%}{Expected\%}\right) \right)$$
    *Thresholds:*
    *   $\text{PSI} < 0.10$: Stable; no action required.
    *   $0.10 \le \text{PSI} < 0.25$: Marginal shift; schedule shadow retraining.
    *   $\text{PSI} \ge 0.25$: Severe drift; alert ML engineers, trigger fallback weights.
2.  **Concept Drift:** Track rolling 30-day F1-scores against delayed confirmation labels (e.g., audit findings or payment clawbacks).

### 8.2 Agentic AI Drift Management
1.  **Behavioral Divergence:** Continuous monitoring of human analyst override rates. A significant dip in the Recommendation Acceptance Rate over a 7-day period triggers a prompt and context index review.
2.  **Factual Consistency Evaluator:** Run an automated verification cycle comparing output JSON summaries with raw database responses. If claims contain ungrounded variables, increment the alert count.
3.  **Prompt & Dependency Drift:** Continuous evaluation against a synthetic benchmark of 50 known historical fraud scenarios to detect regressions following foundation model updates.

---

## 9. Edge Case Analysis & Mitigation Matrix

| Operational Scenario | Inherent Risk | Architectural / Procedural Mitigation |
| :--- | :--- | :--- |
| **Sparse Label Cold-Start** | Inability to train robust supervised classifiers on day one. | Rely on Tier 1 deterministic rules and Tier 2 unsupervised anomaly models (Isolation Forest). Seed features using synthetic fraud simulations (CTGAN). |
| **High OCR Error Rates** | Low-quality scans trigger incorrect duplicate or amount alerts. | Attach field-level extraction confidence flags. Route documents with confidence $< 0.85$ to an async human-in-the-loop validation queue before score generation. |
| **Supplier Name Collisions** | Legitimate companies with near-identical names trigger false link analyses. | Leverage multi-attribute resolution requiring intersection of Tax ID, registration jurisdiction, bank routing, and physical address before establishing graph edges. |
| **Mid-Cycle Bank Modification** | Account takeover or business email compromise diverting funds prior to a run. | Strict policy rule: Freeze invoice execution for 72 hours upon bank details modification. Require out-of-band dual-authorization before unfreezing. |
| **Collusion Across Actors** | Internal procurement officer colluding with an external vendor. | Neo4j/Neptune link-analysis traverses corporate registry filings, shared phone/address data, and historical single-bid award concentrations. |
| **Smurfing (Split Invoicing)** | Splitting large invoices into smaller payments to stay under review limits. | Rolling window aggregation computing cumulative spend grouped by `(Vendor_ID, Cost_Center, Category)` over 24-hour, 7-day, and 30-day windows. |
| **Novel / Zero-Day Fraud Vectors** | Emergent attack variations passing standard rules and classifiers. | Rapid feedback loop: Investigator-labeled cases automatically convert into synthetic test suites and candidate heuristic rules within 24 hours. |
| **Third-Party API Downtime** | Vendor bank or registry verification services go offline. | Circuit breaker safely falls back to cached verifications, logs a degradation event, and temporarily shifts transactions to heightened manual review queues. |

---

## 10. Implementation Governance & Rollout Roadmap

### 10.1 Governance, Compliance & Auditability
*   **Immutable Forensic Logs:** Store all model inputs, intermediate agent rationales, evidence citations, and human decisions within WORM (Write Once, Read Many) object storage for SOX §404 compliance.
*   **Privacy & Data Minimization:** Tokenize and mask personally identifiable information (PII) before transmission to external LLM providers.
*   **Model Governance:** Maintain comprehensive model cards detailing training lineage, demographic parity evaluations, and performance boundaries.

### 10.2 Phased Rollout Roadmap

```
Phase 1: Foundation (Weeks 1–6)
├── Deploy data ingestion pipelines across core ERP systems.
├── Implement Tier 1 Deterministic Rule Engine for immediate risk capture.
└── Establish immutable event bus and audit storage framework.

Phase 2: Advanced Analytics (Weeks 7–14)
├── Deploy feature store and ingest historical transaction data.
├── Train and validate Tier 2 ML models (Supervised + Unsupervised).
├── Build supplier-employee relationship graph in Neo4j/Neptune.
└── Deploy ML inference endpoints in shadow mode.

Phase 3: Agentic Orchestration & Triage (Weeks 15–20)
├── Implement Tier 3 Agentic AI orchestrator and connect LLM pipelines.
├── Configure investigator case management UI and evidence viewers.
└── Establish guardrails, citation enforcement, and HITL approval gates.

Phase 4: Pilot & Operational Validation (Weeks 21–24)
├── Launch targeted pilot covering 15–20% of enterprise transaction volume.
├── Measure precision, recall, false positive rates, and agent acceptance.
├── Fine-tune rule thresholds and model calibration curves.
└── Run drift detection monitors in real-time.

Phase 5: Full Enterprise Scale (Weeks 25+)
├── Scale solution across 100% of global operational procurement volume.
├── Activate automated feedback loops for continuous retraining.
└── Integrate automated SIEM/ERP writebacks for sanctioned actions.
```

---

## Appendix: Automated Triage Output Schema

When the Agentic AI layer compiles a case for human review, it generates a structured, verified schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "case_id": "CASE-2026-88192",
  "timestamp": "2026-09-14T10:15:30Z",
  "transaction_identifiers": {
    "invoice_id": "INV-990214",
    "po_reference": "PO-554102",
    "vendor_id": "VEND-8821"
  },
  "risk_assessment": {
    "composite_risk_score": 0.892,
    "classification": "HIGH_RISK",
    "rule_violations": [
      "BANK_MODIFIED_WITHIN_72_HOURS",
      "THRESHOLD_PROXIMITY_SPLIT_DETECTED"
    ],
    "top_contributing_factors": [
      {
        "feature": "vendor_bank_account_age_days",
        "value": 1.5,
        "shap_value": 0.381
      },
      {
        "feature": "invoice_amount_vs_90d_avg_ratio",
        "value": 4.12,
        "shap_value": 0.245
      }
    ]
  },
  "graph_findings": {
    "entity_conflicts_detected": true,
    "details": "Shared residential address found between Vendor Director and AP Approver [Node ID: EMP-10492]."
  },
  "agent_narrative": "Invoice INV-990214 ($49,850.00) was submitted 36 hours following a bank account modification. The amount sits just below the $50,000 dual-approver threshold. Graph linkage reveals shared contact attributes between the designated approver and vendor leadership. High probability of internal collusion and diverted payment.",
  "recommended_action": "HOLD_PAYMENT_AND_INITIATE_INDEPENDENT_AUDIT",
  "requires_human_signoff": true,
  "evidence_manifest": [
    {
      "source": "ERP_AUDIT_LOG",
      "record_id": "LOG-BANK-CHANGE-88129",
      "verification_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "source": "GRAPH_DB",
      "record_id": "EDGE-DIRECTOR-APPROVER-0012",
      "verification_hash": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    }
  ]
}
```