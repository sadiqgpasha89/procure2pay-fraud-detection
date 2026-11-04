"""Deploy the Tier-3 investigator to Vertex AI Agent Engine (managed agent runtime) instead of Cloud Run.

NOT executed in this repository (needs a GCP project). The API shape was checked against google-cloud-aiplatform 2.7
(`vertexai.agent_engines.create(agent_engine=<ADK BaseAgent>, requirements=..., extra_packages=..., env_vars=...)`).

The specialists (graph-analyst, doc-auditor) and MCP servers keep running on Cloud Run; Agent Engine reaches them over HTTPS using the
same bearer token, or privately via a PSC interface (--psc-network-attachment). Prerequisites:
    uv build --wheel                     # produces dist/p2p_fraud-*.whl
    export GOOGLE_CLOUD_PROJECT=... SERVICE_TOKEN=... GRAPH_ANALYST_URL=https://... DOC_AUDITOR_URL=https://... MCP_ERP_URL=... MCP_KNOWLEDGE_URL=... MCP_GRAPH_URL=...
    python deploy/agent_engine/deploy_investigator.py --staging-bucket gs://PROJECT-p2p-staging
Then point the gateway at it with an Agent-Engine A2A/stream-query client (or keep the Cloud Run investigator as the default path).
"""
from __future__ import annotations

import argparse
import glob
import os

import vertexai
from vertexai import agent_engines

from p2p.agents.apps import remote_agent
from p2p.agents.investigator import InvestigatorAgent
from p2p.common.config import get_settings

ENV_KEYS = ["SERVICE_TOKEN", "GRAPH_ANALYST_URL", "DOC_AUDITOR_URL", "MCP_ERP_URL", "MCP_GRAPH_URL", "MCP_KNOWLEDGE_URL", "LLM_PROVIDER", "GEMINI_MODEL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-bucket", required=True)
    ap.add_argument("--display-name", default="p2p-investigator")
    ap.add_argument("--service-account", default=None)
    ap.add_argument("--min-instances", type=int, default=0)
    a = ap.parse_args()
    s = get_settings()
    vertexai.init(project=s.GOOGLE_CLOUD_PROJECT or os.environ["GOOGLE_CLOUD_PROJECT"], location=s.GOOGLE_CLOUD_LOCATION, staging_bucket=a.staging_bucket)
    agent = InvestigatorAgent(graph_agent=remote_agent("graph_analyst", s.GRAPH_ANALYST_URL, "Graph link analyst (A2A)", minimal_task=True),
                              doc_agent=remote_agent("document_auditor", s.DOC_AUDITOR_URL, "Document & contract auditor (A2A)", minimal_task=True))
    wheel = sorted(glob.glob("dist/p2p_fraud-*.whl"))[-1]
    env = {k: os.environ[k] for k in ENV_KEYS if k in os.environ} | {"LLM_PROVIDER": "vertex"}
    remote = agent_engines.create(
        agent_engine=agent, display_name=a.display_name, requirements=["google-adk[a2a]>=2.13,<3", "fastmcp>=4.1,<5", wheel],
        extra_packages=[wheel], env_vars=env, service_account=a.service_account, min_instances=a.min_instances)
    print("deployed:", remote.resource_name)


if __name__ == "__main__":
    main()
