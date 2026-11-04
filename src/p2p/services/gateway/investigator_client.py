"""Gateway-side A2A client for the investigator agent (ADK RemoteA2aAgent driven by a Runner)."""
from __future__ import annotations

import asyncio
import warnings

import httpx
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from p2p.common.config import get_settings
from p2p.common.models import CaseDossier, InvestigationRequest

warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")


class InvestigationError(RuntimeError):
    pass


class InvestigatorClient:
    def __init__(self) -> None:
        s = get_settings()
        self._timeout = s.INVESTIGATION_TIMEOUT_SEC
        agent = RemoteA2aAgent(name="investigator", description="Tier-3 investigator (A2A)",
                               agent_card=f"{s.INVESTIGATOR_URL.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}",
                               httpx_client=httpx.AsyncClient(timeout=self._timeout, headers=s.auth_headers()))
        self._runner = InMemoryRunner(agent=agent, app_name="p2p-gateway")

    async def investigate(self, req: InvestigationRequest) -> CaseDossier:
        session = await self._runner.session_service.create_session(app_name="p2p-gateway", user_id="gateway")
        last = ""

        async def _run() -> None:
            nonlocal last
            async for ev in self._runner.run_async(user_id="gateway", session_id=session.id,
                                                   new_message=types.Content(role="user", parts=[types.Part(text=req.model_dump_json())])):
                if ev.content and ev.content.parts and ev.content.parts[0].text:
                    last = ev.content.parts[0].text

        try:
            await asyncio.wait_for(_run(), timeout=self._timeout)
            return CaseDossier.model_validate_json(last)
        except Exception as exc:  # timeouts, A2A transport errors, malformed dossier
            raise InvestigationError(f"{type(exc).__name__}: {exc}") from exc

    async def warm(self) -> None:
        """Resolve the agent card and open the A2A transport before traffic arrives. A malformed request makes the investigator
        answer immediately without touching the specialists, so warming costs ~nothing and creates no case."""
        session = await self._runner.session_service.create_session(app_name="p2p-gateway", user_id="warmup")
        async for _ in self._runner.run_async(user_id="warmup", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text="warmup")])):
            pass
