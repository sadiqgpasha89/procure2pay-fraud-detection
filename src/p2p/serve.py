"""Process launcher: `python -m p2p.serve <service> [--port N]`.  One image, many services (Cloud Run: set CMD per service)."""
from __future__ import annotations

import argparse
import os

import uvicorn

# service → (ASGI import path, default port)
SERVICES: dict[str, tuple[str, int]] = {
    "gateway": ("p2p.services.gateway.app:app", 8100),
    "scoring": ("p2p.services.scoring.app:app", 8101),
    "case": ("p2p.services.case.app:app", 8102),
    "audit": ("p2p.services.audit.app:app", 8103),
    "mcp-erp": ("p2p.mcp_servers.erp_server:app", 8201),
    "mcp-graph": ("p2p.mcp_servers.graph_server:app", 8202),
    "mcp-knowledge": ("p2p.mcp_servers.knowledge_server:app", 8203),
    "investigator": ("p2p.agents.apps:investigator_app", 8300),
    "graph-analyst": ("p2p.agents.apps:graph_analyst_app", 8301),
    "doc-auditor": ("p2p.agents.apps:doc_auditor_app", 8302),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("service", choices=sorted(SERVICES))
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "0")))
    a = ap.parse_args()
    target, default_port = SERVICES[a.service]
    uvicorn.run(target, host=a.host, port=a.port or default_port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
