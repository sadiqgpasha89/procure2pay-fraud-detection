"""Prompt Template Registry with versioning, security boundaries, and parameter substitution."""
from functools import lru_cache
from typing import Dict, Optional
from pydantic import BaseModel


class PromptTemplate(BaseModel):
    name: str
    version: str
    system_prompt: str
    template: str
    description: str


GRAPH_ANALYST_SYSTEM = """You are an expert Forensic Graph Intelligence Analyst for enterprise Procure-to-Pay fraud prevention.
Analyze graph linkages for shared phone numbers, residential addresses, and single-bid awards between vendor leadership and internal approvers.
Strict Rule: Do not emit SQL or state-changing ERP calls. Every claim must refer to verifiable graph edge identifiers."""

DOC_AUDITOR_SYSTEM = """You are a Senior Internal Audit and 3-Way Reconciliation Specialist.
Audit invoice attachments, purchase orders, and goods receipts for price inflation, duplicate billings, and micro-variance exploitation.
Strict Rule: Do not hallucinate delivery notes or quantities. Cite source ERP document IDs."""

RISK_SYNTHESIZER_SYSTEM = """You are the Chief Enterprise Fraud Investigator Synthesizer.
Synthesize all specialist findings into the official JSON Case Synthesis Dossier.
Strict Rules:
1. Every claim must cite verified evidence in format: [Evidence: <SOURCE>].
2. Output strictly adhering to the JSON schema.
3. Recommended action must be strongly typed."""


class PromptRegistry:
    """Registry maintaining versioned system and task prompts."""

    def __init__(self):
        self.prompts: Dict[str, PromptTemplate] = {}
        self._init_default_prompts()

    def _init_default_prompts(self):
        self.register(
            PromptTemplate(
                name="graph_link_analyst",
                version="1.0.0",
                system_prompt=GRAPH_ANALYST_SYSTEM,
                template="Investigate vendor {vendor_id} and approver {approver_id}. Graph findings: {graph_findings}",
                description="Specialist prompt for inspecting collusion paths in Neo4j/knowledge graph."
            )
        )
        self.register(
            PromptTemplate(
                name="document_auditor",
                version="1.0.0",
                system_prompt=DOC_AUDITOR_SYSTEM,
                template="Audit invoice {invoice_id} (${amount:.2f}) against PO {po_id} and GR {gr_id}.",
                description="Specialist prompt for 3-way reconciliation."
            )
        )
        self.register(
            PromptTemplate(
                name="risk_synthesizer",
                version="1.0.0",
                system_prompt=RISK_SYNTHESIZER_SYSTEM,
                template="Synthesize case for invoice {invoice_id}, score {risk_score:.3f}, violations: {violations}.",
                description="Synthesizer prompt for generating canonical Case Dossier."
            )
        )

    def register(self, template: PromptTemplate):
        self.prompts[f"{template.name}:{template.version}"] = template

    def get_prompt(self, name: str, version: str = "1.0.0") -> Optional[PromptTemplate]:
        return self.prompts.get(f"{name}:{version}")


@lru_cache()
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
