"""Investigator — the Tier-3 orchestrator (ADK custom agent, exposed over A2A).

Flow (deterministic skeleton, LLM only where language/judgement is needed):
  1. fan out in PARALLEL to remote specialists over A2A: graph_analyst ∥ document_auditor
  2. parse their typed reports; RE-VERIFY every cited evidence hash against the owning MCP server
     → fabricated or stale citations are dropped, claims left without evidence are dropped
  3. narrator LLM writes the prose from verified findings only
  4. guardrails: citation + grounding check (fallback to a deterministic template on violation)
  5. deterministic policy picks the recommended action; dossier is emitted, always requiring human sign-off
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import structlog
from fastmcp import Client
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.genai import types

from p2p.agents import prompts
from p2p.agents.llm import build_model
from p2p.agents.narrative import check_narrative, template_narrative
from p2p.agents.policy import decide_action
from p2p.agents.schemas import Finding, SpecialistReport, parse_report_ex
from p2p.common.config import get_settings
from p2p.common.models import (
    CaseDossier, Evidence, GraphFindings, InvestigationRequest, RiskAssessment, TransactionIdentifiers,
)

log = structlog.get_logger(__name__)
GRAPH_KINDS = {"STAFF_VENDOR_LINK", "SHARED_BANK_ACCOUNT", "SHARED_ADDRESS_OR_PHONE"}


class EvidenceVerifier:
    """Asks the MCP server that minted a citation to re-derive its hash. Unknown source or any error ⇒ invalid."""

    def __init__(self) -> None:
        s = get_settings()
        self.urls = {"ERP": s.MCP_ERP_URL, "GRAPH_DB": s.MCP_GRAPH_URL, "KNOWLEDGE_BASE": s.MCP_KNOWLEDGE_URL}
        self.token = s.SERVICE_TOKEN or None

    async def verify(self, items: list[Evidence]) -> dict[tuple[str, str], bool]:
        out: dict[tuple[str, str], bool] = {}
        by_source: dict[str, list[Evidence]] = {}
        for e in items:
            by_source.setdefault(e.source, []).append(e)
        for source, evs in by_source.items():
            url = self.urls.get(source)
            if not url:
                out.update({(e.source, e.record_id): False for e in evs})
                continue
            try:
                async with Client(url, auth=self.token) as c:
                    for e in evs:
                        r = await c.call_tool("verify_evidence", {"record_id": e.record_id, "verification_hash": e.verification_hash})
                        out[(e.source, e.record_id)] = bool((r.data or {}).get("valid"))
            except Exception as exc:  # fail closed
                log.error("evidence_verify_failed", source=source, error=str(exc))
                out.update({(e.source, e.record_id): False for e in evs})
        return out


class InvestigatorAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    specialists: ParallelAgent
    narrator: LlmAgent
    verifier: EvidenceVerifier

    def __init__(self, graph_agent: BaseAgent, doc_agent: BaseAgent, name: str = "investigator", verifier: EvidenceVerifier | None = None):
        specialists = ParallelAgent(name="specialists", sub_agents=[graph_agent, doc_agent])
        narrator = LlmAgent(
            name="narrator", model=build_model("narrator"), description="Writes the cited investigation narrative from verified findings.",
            instruction=lambda ctx: _narrator_instruction(ctx),
            generate_content_config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=700))
        super().__init__(name=name, description="Coordinates graph + document specialists over A2A and compiles the case dossier.",
                         specialists=specialists, narrator=narrator, verifier=verifier or EvidenceVerifier(), sub_agents=[specialists, narrator])

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.perf_counter()
        text = " ".join(p.text for p in (ctx.user_content.parts if ctx.user_content else []) if p.text)
        try:
            req = InvestigationRequest.model_validate_json(text[text.index("{"):text.rindex("}") + 1])
        except Exception as exc:
            yield self._final(ctx, json.dumps({"error": f"invalid investigation request: {exc}"}))
            return
        notes: list[str] = []

        # 1) fan out to specialists (A2A) ----------------------------------------------------------
        reports: dict[str, SpecialistReport | None] = {"graph_analyst": None, "document_auditor": None}
        async for ev in self.specialists.run_async(ctx):
            if ev.author in reports and ev.is_final_response() and ev.content and ev.content.parts:
                raw = "".join(p.text or "" for p in ev.content.parts)
                rep, problems = parse_report_ex(raw, ev.author)
                for pr in problems:
                    log.warning("specialist_report_issue", agent=ev.author, issue=pr, preview=raw[:300])
                    notes.append(f"{ev.author}: {pr}")
                reports[ev.author] = rep or reports[ev.author]
            yield ev
        incomplete = False
        for name, rep in reports.items():
            if rep is None:
                incomplete = True
                notes.append(f"{name} returned no valid report; its findings are absent from this dossier.")
            elif rep.summary.startswith("TOOL_ERRORS"):
                incomplete = True
                notes.append(f"{name} reported tool failures ({rep.summary[12:].strip()}); evidence is incomplete.")

        # 2) verify evidence ------------------------------------------------------------------------
        findings = [f for rep in reports.values() if rep for f in rep.findings]
        valid = await self.verifier.verify(list({(e.source, e.record_id, e.verification_hash): e for f in findings for e in f.evidence}.values()))
        verified: list[Finding] = []
        for f in findings:
            good = [e for e in f.evidence if valid.get((e.source, e.record_id))]
            if len(good) < len(f.evidence):
                notes.append(f"Dropped {len(f.evidence) - len(good)} unverifiable citation(s) from claim '{f.claim[:60]}…'.")
            if f.kind == "NO_ANOMALY" or good:
                verified.append(f.model_copy(update={"evidence": good}))
            else:
                notes.append(f"Dropped ungrounded claim: '{f.claim[:80]}…'.")
        manifest = list({(e.source, e.record_id): e for f in verified for e in f.evidence}.values())

        # 3) narrative ------------------------------------------------------------------------------
        t1, t2 = req.scoring.tier1, req.scoring.tier2
        facts = {"invoice_id": req.invoice_id, "vendor_id": req.vendor_id, "amount": req.amount,
                 "score": t2.composite_score if t2 else 0.0, "classification": t2.classification.value if t2 else "BLOCKED",
                 "rule_violations": t1.violations}
        fjson = [f.model_dump(mode="json") for f in verified]
        yield Event(invocation_id=ctx.invocation_id, author=self.name,
                    actions=EventActions(state_delta={"facts": json.dumps(facts), "verified_findings": json.dumps(fjson)}))
        narrative = ""
        async for ev in self.narrator.run_async(ctx):
            if ev.is_final_response() and ev.content and ev.content.parts:
                narrative = "".join(p.text or "" for p in ev.content.parts).strip()
        problems = check_narrative(narrative, manifest, facts, fjson) if narrative else ["empty narrative"]
        if problems:
            notes.append(f"Narrative failed guardrails ({'; '.join(problems[:3])}); replaced with deterministic template.")
            narrative = template_narrative(facts, fjson)

        # 4) deterministic action + dossier -----------------------------------------------------------
        action, why = decide_action(t1, t2, verified, incomplete)
        graph_f = [f for f in verified if f.kind in GRAPH_KINDS]
        nodes = sorted({n for f in graph_f for n in re.findall(r"\b(?:EMP|VEND)-\d+\b", f.claim)} - {req.vendor_id})
        dossier = CaseDossier(
            case_id=req.case_id, timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            transaction_identifiers=TransactionIdentifiers(invoice_id=req.invoice_id, vendor_id=req.vendor_id, po_reference=req.po_reference),
            risk_assessment=RiskAssessment(composite_risk_score=t2.composite_score if t2 else 1.0, classification=t2.classification if t2 else "HIGH_RISK",
                                           rule_violations=t1.violations, top_contributing_factors=(t2.contributions[:3] if t2 else [])),
            graph_findings=GraphFindings(entity_conflicts_detected=bool(graph_f), collusion_nodes=nodes,
                                         details="; ".join(f.claim for f in graph_f) or "No conflicting graph linkages detected."),
            document_findings=[f.claim for f in verified if f.kind not in GRAPH_KINDS and f.kind != "NO_ANOMALY"],
            agent_narrative=narrative, recommended_action=action, requires_human_signoff=True, evidence_manifest=manifest,
            guardrail_notes=notes + [f"Action rationale: {why}"],
            llm_provider=f"{get_settings().LLM_PROVIDER}|prompts@{prompts.prompt_fingerprint()}", synthesis_latency_sec=round(time.perf_counter() - t0, 3))
        yield self._final(ctx, dossier.model_dump_json(by_alias=True))

    def _final(self, ctx: InvocationContext, text: str) -> Event:
        return Event(invocation_id=ctx.invocation_id, author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))


def _narrator_instruction(ctx: ReadonlyContext) -> str:
    st = ctx.state
    return f"{prompts.NARRATOR}\nFACTS: {st.get('facts', '{}')}\nVERIFIED_FINDINGS: {st.get('verified_findings', '[]')}\n"


def new_case_id() -> str:
    return f"CASE-{datetime.now(timezone.utc).year}-{uuid.uuid4().hex[:6].upper()}"
