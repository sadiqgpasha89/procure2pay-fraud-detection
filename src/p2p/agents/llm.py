"""Model factory: one switch (`LLM_PROVIDER`) moves Tier 3 between free/offline, open-weights and Vertex AI Gemini."""
from __future__ import annotations

import os

from google.adk.models.base_llm import BaseLlm

from p2p.agents.reference_llm import ReferenceLlm
from p2p.common.config import get_settings


def build_model(role: str) -> BaseLlm | str:
    s = get_settings()
    if s.LLM_PROVIDER == "offline":
        return ReferenceLlm(role=role)
    if s.LLM_PROVIDER == "ollama":
        from google.adk.models.lite_llm import LiteLlm  # needs `pip install .[ollama]`

        return LiteLlm(model=s.OLLAMA_MODEL, api_base=s.OLLAMA_API_BASE, think=s.OLLAMA_THINK)
    # Vertex AI Gemini: ADK resolves the string; credentials come from ADC / the Cloud Run service account.
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    if s.GOOGLE_CLOUD_PROJECT:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", s.GOOGLE_CLOUD_PROJECT)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", s.GOOGLE_CLOUD_LOCATION)
    return s.GEMINI_MODEL
