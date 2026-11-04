"""ASGI entrypoints for the three agent microservices. Each is an A2A server (agent card at /.well-known/agent-card.json)."""
from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import BaseAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent
from starlette.responses import JSONResponse

from google.genai import types

from p2p.agents.investigator import InvestigatorAgent
from p2p.agents.specialists import build_document_auditor, build_graph_analyst
from p2p.common.config import get_settings
from p2p.common.models import InvestigationRequest
from p2p.common.ops import BearerAuthMiddleware, configure_logging, metrics_endpoint


def _specialist_task(ctx, agent_name, converter):
    """Outgoing A2A message for a specialist: ONLY the identifiers. Specialists must corroborate from source systems through their tools;
    they never see the Tier-2 score, SHAP drivers or features (avoids anchoring on the ML output and keeps their evidence independent)."""
    text = " ".join(p.text for p in (ctx.user_content.parts if ctx.user_content else []) if p.text)
    req = InvestigationRequest.model_validate_json(text[text.index("{"):text.rindex("}") + 1])
    task = {"task": "independently corroborate this invoice from source systems", "case_id": req.case_id, "invoice_id": req.invoice_id, "vendor_id": req.vendor_id}
    parts = converter(types.Part(text=json.dumps(task)))
    return (list(parts) if isinstance(parts, (list, tuple)) else [parts]), None


def remote_agent(name: str, base_url: str, description: str, minimal_task: bool = False) -> RemoteA2aAgent:
    """Client-side proxy for an agent running in another process (A2A over HTTP, JSON-RPC)."""
    client = httpx.AsyncClient(timeout=get_settings().INVESTIGATION_TIMEOUT_SEC, headers=get_settings().auth_headers())
    return RemoteA2aAgent(name=name, description=description, agent_card=f"{base_url.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}", httpx_client=client,
                          context_builder=_specialist_task if minimal_task else None)


def _serve(agent: BaseAgent, public_url: str, service: str):
    configure_logging(service)
    u = urlparse(public_url)
    app = to_a2a(agent, host=u.hostname or "localhost", port=u.port or (443 if u.scheme == "https" else 80), protocol=u.scheme or "http")
    app.add_route("/health", lambda _: JSONResponse({"status": "ok", "service": service}))
    app.add_route("/metrics", metrics_endpoint)
    return BearerAuthMiddleware(app)


def build_graph_analyst_app():
    return _serve(build_graph_analyst(), get_settings().GRAPH_ANALYST_URL, "graph-analyst")


def build_doc_auditor_app():
    return _serve(build_document_auditor(), get_settings().DOC_AUDITOR_URL, "doc-auditor")


def build_investigator_app():
    s = get_settings()
    agent = InvestigatorAgent(
        graph_agent=remote_agent("graph_analyst", s.GRAPH_ANALYST_URL, "Graph link analyst (A2A)", minimal_task=True),
        doc_agent=remote_agent("document_auditor", s.DOC_AUDITOR_URL, "Document & contract auditor (A2A)", minimal_task=True))
    return _serve(agent, s.INVESTIGATOR_URL, "investigator")


# module-level ASGI apps are built lazily so importing this module (e.g. in tests) has no side effects
def __getattr__(name: str):
    builders = {"graph_analyst_app": build_graph_analyst_app, "doc_auditor_app": build_doc_auditor_app, "investigator_app": build_investigator_app}
    if name in builders:
        app = builders[name]()
        globals()[name] = app
        return app
    raise AttributeError(name)


