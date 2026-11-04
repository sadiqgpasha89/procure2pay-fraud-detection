"""Specialist agents (ADK LlmAgent + MCP toolsets). Each runs as its own A2A microservice."""
from __future__ import annotations

from collections import defaultdict

import structlog
from google.adk.agents import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types

from p2p.agents import prompts
from p2p.agents.llm import build_model
from p2p.agents.reference_llm import INV_RE, ReferenceLlm, unwrap
from p2p.common.config import get_settings

log = structlog.get_logger(__name__)

GRAPH_TOOLS = ["get_entity_links"]
DOC_TOOLS = ["get_transaction_context", "get_bank_change_log", "find_related_invoices", "get_vendor_invoice_history", "search_documents"]


def _toolset(url: str, allow: list[str]) -> McpToolset:
    return McpToolset(connection_params=StreamableHTTPConnectionParams(url=url, headers=get_settings().auth_headers(), timeout=15.0),
                      tool_filter=allow)  # allow-list: verify_evidence is intentionally NOT exposed to the model


def _audit_tool_call(allow: list[str]):
    """before_tool_callback: structured audit of every tool the model invokes + hard allow-list (defence in depth)."""
    def cb(tool, args, tool_context):
        if tool.name not in allow:
            log.warning("tool_blocked", tool=tool.name)
            return {"error": f"tool '{tool.name}' is not permitted for this agent"}
        log.info("agent_tool_call", agent=tool_context.agent_name, tool=tool.name, args=args)
        return None
    return cb


def _prefetch(role: str):
    """before_model_callback: issue the deterministic first-hop tool calls ourselves (see config.TOOL_PREFETCH). Planning logic is the
    reference model's planning half, so the offline path and the prefetch path can never drift apart. Once the evidence is in the
    context, return None and let the real model write the report."""
    plan = getattr(ReferenceLlm(role=role), f"_{role}")

    def cb(callback_context, llm_request):
        responses: dict[str, list[dict]] = defaultdict(list)
        user_text = ""
        for c in llm_request.contents or []:
            for p in c.parts or []:
                if p.function_response:
                    responses[p.function_response.name].append(unwrap(p.function_response.response))
                if p.text and c.role == "user":
                    user_text += " " + p.text
        found = INV_RE.findall(user_text)
        calls, _ = plan(found[0] if found else "", responses, "")
        if not calls:
            return None
        log.info("tool_prefetch", agent=role, tools=[n for n, _ in calls])
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]))
    return cb


def _use_prefetch() -> bool:
    s = get_settings()
    return s.TOOL_PREFETCH == "on" or (s.TOOL_PREFETCH == "auto" and s.LLM_PROVIDER == "ollama")


def _agent(name: str, description: str, instruction: str, tools: list[McpToolset], allow: list[str]) -> LlmAgent:
    return LlmAgent(name=name, description=description, model=build_model(name), instruction=instruction, tools=tools,
                    before_model_callback=_prefetch(name) if (_use_prefetch() and get_settings().LLM_PROVIDER != "offline") else None,
                    before_tool_callback=_audit_tool_call(allow),
                    generate_content_config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096))


def build_graph_analyst() -> LlmAgent:
    return _agent("graph_analyst", "Finds shared bank accounts, addresses and phone numbers linking a vendor to other vendors or to staff on the transaction.",
                  prompts.GRAPH_ANALYST, [_toolset(get_settings().MCP_GRAPH_URL, GRAPH_TOOLS)], GRAPH_TOOLS)


def build_document_auditor() -> LlmAgent:
    s = get_settings()
    return _agent("document_auditor", "Audits an invoice against its PO, goods receipt, bank-change history, related invoices, contract terms and e-mails (RAG).",
                  prompts.DOCUMENT_AUDITOR,
                  [_toolset(s.MCP_ERP_URL, DOC_TOOLS[:4]), _toolset(s.MCP_KNOWLEDGE_URL, ["search_documents"])], DOC_TOOLS)
