"""Deterministic reference "model" implementing the same tool-use contract as a real LLM.

Why it exists: (1) the full ADK + MCP + A2A stack runs in CI and on a laptop with no API key and no GPU;
(2) it is the regression baseline the benchmark compares real models against; (3) it keeps the platform
operational (plainer narrative, identical safety properties) if the LLM provider is down.
It issues real tool calls through ADK, so everything downstream of the model is exercised for real.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from p2p.agents.narrative import template_narrative
from p2p.agents.schemas import Finding, SpecialistReport
from p2p.common.models import Evidence

ATTR_WORD = {"ADDR": "address", "PHONE": "phone number", "BANK": "bank account"}
INV_RE = re.compile(r"INV-\d{6}")
Responses = dict[str, list[dict]]  # tool name → payloads, in call order


def unwrap(resp: Any) -> dict:
    """MCP tool results arrive as {content:[{text}], structuredContent:{...}} — return the payload dict."""
    if isinstance(resp, dict):
        if resp.get("isError"):
            return {"error": " ".join(str(c.get("text", "")) for c in resp.get("content", []) if isinstance(c, dict))[:300] or "tool error"}
        if isinstance(resp.get("structuredContent"), dict):
            return resp["structuredContent"]
        for part in resp.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "text":
                try:
                    return json.loads(part["text"])
                except (ValueError, TypeError):
                    pass
        if isinstance(resp.get("result"), dict):
            return resp["result"]
    return resp if isinstance(resp, dict) else {}


def _evidence(items: list[dict], ids: set[str]) -> list[Evidence]:
    return [Evidence(**e) for e in items if e.get("record_id") in ids]


def _evidence_prefix(items: list[dict], *prefixes: str) -> list[Evidence]:
    return [Evidence(**e) for e in items if e.get("record_id", "").startswith(prefixes)]


def _marker_json(system: str, marker: str) -> Any:
    i = system.find(marker)
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(system[i + len(marker):].lstrip())[0]
    except ValueError:
        return None


def _system_text(req: LlmRequest) -> str:
    si = getattr(req.config, "system_instruction", None)
    if isinstance(si, types.Content):
        return " ".join(p.text for p in si.parts or [] if p.text)
    return str(si or "")


class ReferenceLlm(BaseLlm):
    model: str = "p2p-reference"
    role: str = "graph_analyst"  # graph_analyst | document_auditor | narrator

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        responses: Responses = defaultdict(list)
        user_text = ""
        for c in llm_request.contents or []:
            for p in c.parts or []:
                if p.function_response:
                    responses[p.function_response.name].append(unwrap(p.function_response.response))
                if p.text and c.role == "user":
                    user_text += " " + p.text
        found = INV_RE.findall(user_text)
        calls, text = getattr(self, f"_{self.role}")(found[0] if found else "", responses, _system_text(llm_request))
        parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls] if calls else [types.Part(text=text)]
        yield LlmResponse(content=types.Content(role="model", parts=parts))

    # ── graph analyst ───────────────────────────────────────────────────────
    def _graph_analyst(self, invoice_id: str, r: Responses, _: str):
        if not r["get_entity_links"]:
            return [("get_entity_links", {"invoice_id": invoice_id})], ""
        g = r["get_entity_links"][0]
        if g.get("error"):
            return None, SpecialistReport(agent="graph_analyst", summary="TOOL_ERRORS: get_entity_links").model_dump_json()
        findings: list[Finding] = []
        for link in g.get("links", []):
            ev, who = _evidence(g.get("evidence", []), {link["record_id"]}), link["linked_entity"]
            if link["linked_entity_kind"] == "EMPLOYEE":
                roles = ", ".join(link.get("roles_on_this_transaction") or [])
                findings.append(Finding(
                    kind="STAFF_VENDOR_LINK", severity="HIGH" if roles else "MEDIUM", evidence=ev,
                    claim=f"Vendor {g['vendor_id']} shares an identical {ATTR_WORD[link['attribute_type']]} with employee {who} ({link.get('linked_employee_role')})"
                          + (f" who acted on this transaction as {roles}." if roles else ".")))
            elif link["attribute_type"] == "BANK":
                findings.append(Finding(kind="SHARED_BANK_ACCOUNT", severity="HIGH", evidence=ev,
                                        claim=f"Vendor {g['vendor_id']} uses the same bank account as vendor {who}."))
            else:
                findings.append(Finding(kind="SHARED_ADDRESS_OR_PHONE", severity="MEDIUM", evidence=ev,
                                        claim=f"Vendor {g['vendor_id']} shares an identical {ATTR_WORD[link['attribute_type']]} with vendor {who}."))
        if not findings:
            findings.append(Finding(kind="NO_ANOMALY", severity="INFO", claim=f"No shared bank, address or phone links found for vendor {g.get('vendor_id')}."))
        return None, SpecialistReport(agent="graph_analyst", summary=g.get("summary", ""), findings=findings).model_dump_json()

    # ── document auditor ────────────────────────────────────────────────────
    def _document_auditor(self, invoice_id: str, r: Responses, _: str):
        if not r["get_transaction_context"]:  # hop 1: everything keyed by invoice id, plus company-wide policy
            return [("get_transaction_context", {"invoice_id": invoice_id}), ("get_bank_change_log", {"invoice_id": invoice_id}),
                    ("find_related_invoices", {"invoice_id": invoice_id}), ("get_vendor_invoice_history", {"invoice_id": invoice_id}),
                    ("search_documents", {"query": "segregation of duties splitting invoices bank change procedure three-way match",
                                          "doc_types": ["POLICY"], "k": 2})], ""
        ctx = r["get_transaction_context"][0]
        if ctx.get("error"):
            return None, SpecialistReport(agent="document_auditor", summary="TOOL_ERRORS: get_transaction_context").model_dump_json()
        vendor_id = (ctx.get("invoice") or {}).get("vendor_id", "")
        if len(r["search_documents"]) < 2:  # hop 2: vendor-scoped retrieval needs the vendor id learned in hop 1
            return [("search_documents", {"query": "unit price cap pricing award sole source agreement", "vendor_id": vendor_id, "doc_types": ["CONTRACT"], "k": 2}),
                    ("search_documents", {"query": "bank account change request urgent remittance details", "vendor_id": vendor_id, "doc_types": ["EMAIL"], "k": 3}),
                    ("search_documents", {"query": "delivery note short delivery received", "vendor_id": vendor_id, "doc_types": ["DOCKET"], "k": 2})], ""
        return None, self._audit_report(ctx, r).model_dump_json()

    def _audit_report(self, ctx: dict, r: Responses) -> SpecialistReport:
        f: list[Finding] = []
        inv, match, ev = ctx.get("invoice") or {}, ctx.get("three_way_match") or {}, ctx.get("evidence", [])
        iid, po_id = inv.get("invoice_id", ""), (ctx.get("purchase_order") or {}).get("po_id", "")
        if match.get("po_present") and not match.get("goods_receipt_present"):
            f.append(Finding(kind="NO_GOODS_RECEIPT", severity="HIGH", evidence=_evidence_prefix(ev, "PO:", "INV:"),
                             claim=f"Invoice {iid} has no goods receipt on file for PO {po_id}."))
        elif match.get("goods_receipt_coverage_pct") is not None and match["goods_receipt_coverage_pct"] < 90:
            f.append(Finding(kind="SHORT_DELIVERY", severity="HIGH", evidence=_evidence_prefix(ev, "GR:", "INV:"),
                             claim=f"Goods receipt covers only {match['goods_receipt_coverage_pct']}% of invoice {iid}."))
        if (match.get("po_variance_pct") or 0) > 1.0:
            f.append(Finding(kind="PO_VARIANCE", severity="HIGH" if match["po_variance_pct"] >= 10 else "MEDIUM", evidence=_evidence_prefix(ev, "PO:", "INV:"),
                             claim=f"Invoice {iid} exceeds its PO by {match['po_variance_pct']}%."))
        if match.get("segregation_of_duties_violation"):
            f.append(Finding(kind="SOD_VIOLATION", severity="HIGH", evidence=_evidence_prefix(ev, "PO:"),
                             claim=f"PO {po_id} was created and approved by the same employee {match.get('po_created_by')}."))
        bank = (r["get_bank_change_log"] or [{}])[0]
        for ch in bank.get("changes", []):
            bev = _evidence(bank.get("evidence", []), {f"BANKCHG:{ch['change_id']}"})
            if ch.get("requested_via") == "EMAIL" and ch.get("requested_from") and ch["requested_from"] != bank.get("registered_vendor_domain"):
                f.append(Finding(kind="BANK_CHANGE_EMAIL_LOOKALIKE", severity="HIGH", evidence=bev,
                                 claim=f"Bank details changed {ch['hours_before_invoice']}h before invoice {iid} on an e-mail request from {ch['requested_from']}, "
                                       f"which differs from the registered vendor domain {bank.get('registered_vendor_domain')}."))
            elif ch.get("hours_before_invoice", 1e9) <= 14 * 24:
                f.append(Finding(kind="BANK_CHANGE_RECENT", severity="MEDIUM", evidence=bev,
                                 claim=f"Bank details changed {ch['hours_before_invoice']}h before invoice {iid}."))
        rel = (r["find_related_invoices"] or [{}])[0]
        dups = (rel.get("same_invoice_number") or []) + (rel.get("same_amount_within_7d") or [])
        if dups:
            f.append(Finding(kind="DUPLICATE_INVOICE", severity="HIGH", evidence=_evidence(rel.get("evidence", []), {f"INV:{d['invoice_id']}" for d in dups}),
                             claim=f"Invoice {iid} duplicates earlier invoice(s) {', '.join(d['invoice_id'] for d in dups)} (same number or same amount within 7 days)."))
        comp = rel.get("companions_within_24h_same_cost_center") or []
        if comp and rel.get("combined_24h_total_including_current", 0) >= rel.get("dual_approval_limit", 50_000.0):
            f.append(Finding(kind="SPLIT_COMPANIONS", severity="HIGH", evidence=_evidence(rel.get("evidence", []), {f"INV:{c['invoice_id']}" for c in comp}),
                             claim=f"{len(comp)} companion invoice(s) within 24h bring the combined total to ${rel['combined_24h_total_including_current']:,.2f}, "
                                   f"over the $50,000 dual-approval limit."))
        hist = (r["get_vendor_invoice_history"] or [{}])[0]
        s = hist.get("summary") or {}
        if (s.get("current_vs_mean_ratio") or 0) >= 2.5 and s.get("prior_invoice_count", 0) >= 3:
            f.append(Finding(kind="AMOUNT_OUTLIER", severity="MEDIUM", evidence=_evidence_prefix(ev, "INV:"),
                             claim=f"Invoice {iid} is {s['current_vs_mean_ratio']}x this vendor's mean over {s['prior_invoice_count']} prior invoices."))
        f += self._rag_findings(inv, r["search_documents"])
        failed = sorted(name for name, outs in r.items() for o in outs if o.get("error"))
        if not f and not failed:
            f.append(Finding(kind="NO_ANOMALY", severity="INFO", claim=f"Three-way match, bank history and contract terms for invoice {iid} show no anomaly."))
        return SpecialistReport(agent="document_auditor", summary=f"TOOL_ERRORS: {', '.join(failed)}" if failed else "", findings=f)

    @staticmethod
    def _rag_findings(inv: dict, searches: list[dict]) -> list[Finding]:
        out, line = [], (inv.get("lines") or [{}])[0]
        for res in searches:
            for h in res.get("hits", []):
                ev = _evidence(res.get("evidence", []), {h["record_id"]})
                if h.get("injection_suspected"):
                    out.append(Finding(kind="PROMPT_INJECTION_DETECTED", severity="HIGH", evidence=ev,
                                       claim=f"Document {h['record_id']} contains instruction-like text aimed at automated reviewers (removed before analysis)."))
                if h["doc_type"] != "CONTRACT":
                    continue
                cap = re.search(rf"{re.escape(line.get('item_code', '@'))}[^$]*price cap of \$([\d.,]+)", h["text"])
                if cap and line.get("unit_price") and line["unit_price"] > float(cap.group(1).replace(",", "")) * 1.02:
                    out.append(Finding(kind="CONTRACT_PRICE_EXCEEDED", severity="HIGH", evidence=ev,
                                       claim=f"Unit price ${line['unit_price']:,.2f} for {line['item_code']} exceeds the contractual cap of ${cap.group(1)} ({h['record_id']})."))
                if "No master supply agreement" in h["text"]:
                    out.append(Finding(kind="NO_CONTRACT", severity="MEDIUM", evidence=ev, claim=f"No master supply agreement is on file ({h['record_id']})."))
                if "sole-source" in h["text"]:
                    out.append(Finding(kind="SOLE_SOURCE_AWARD", severity="MEDIUM", evidence=ev,
                                       claim=f"Vendor was awarded sole-source without competitive bidding ({h['record_id']})."))
        return out

    # ── narrator ────────────────────────────────────────────────────────────
    def _narrator(self, invoice_id: str, r: Responses, system: str):
        return None, template_narrative(_marker_json(system, "FACTS:") or {}, _marker_json(system, "VERIFIED_FINDINGS:") or [])
