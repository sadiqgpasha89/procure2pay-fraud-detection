"""Cross-cutting service plumbing: logging, auth, metrics, circuit breaker, HTTP client."""
from __future__ import annotations

import hmac
import logging
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

import httpx
import structlog
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from p2p.common.config import get_settings


# ── logging ──────────────────────────────────────────────────────────────────
def configure_logging(service: str) -> None:
    level = getattr(logging, get_settings().LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service)


# ── service-to-service auth ──────────────────────────────────────────────────
OPEN_PATHS = ("/health", "/metrics", "/.well-known/")


class BearerAuthMiddleware:
    """Pure-ASGI shared-secret check. In production Cloud Run IAM (OIDC ID tokens) replaces it:
    deploy services with `--no-allow-unauthenticated` and grant `roles/run.invoker` per caller."""

    def __init__(self, app: Callable, token: str | None = None):
        self.app = app
        self.token = token if token is not None else get_settings().SERVICE_TOKEN

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http" and self.token:
            path = scope.get("path", "")
            if not path.startswith(OPEN_PATHS):
                headers = dict(scope.get("headers") or [])
                supplied = headers.get(b"authorization", b"").decode()
                if not hmac.compare_digest(supplied, f"Bearer {self.token}"):
                    resp = JSONResponse({"detail": "unauthorized"}, status_code=401)
                    await resp(scope, receive, send)
                    return
        await self.app(scope, receive, send)


# ── metrics ──────────────────────────────────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "p2p_request_latency_seconds", "Request latency", ["service", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
DECISIONS = Counter("p2p_decisions_total", "Decisions by route/action", ["route", "action"])


async def metrics_endpoint(_: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── circuit breaker (Resilience4j-style) ─────────────────────────────────────
class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Guards a flaky dependency. While not CLOSED, `risk_penalty()` is added to the Tier-2 score
    (spec §6: graceful degradation with S_penalty = +0.15)."""

    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: float = 30.0, penalty: float = 0.15):
        self.name, self.failure_threshold, self.reset_timeout, self.penalty = name, failure_threshold, reset_timeout, penalty
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.degradations = 0

    def call(self, fn: Callable[..., Any], *args: Any, fallback: Callable[..., Any] | None = None, **kw: Any) -> Any:
        if self.state is CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                return self._degrade(fallback, *args, **kw)
        try:
            out = fn(*args, **kw)
        except Exception:
            self.failures += 1
            if self.state is CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state, self.opened_at = CircuitState.OPEN, time.monotonic()
            return self._degrade(fallback, *args, **kw, _reraise=True)
        self.failures, self.state = 0, CircuitState.CLOSED
        return out

    def _degrade(self, fallback: Callable[..., Any] | None, *args: Any, _reraise: bool = False, **kw: Any) -> Any:
        self.degradations += 1
        if fallback is None:
            if _reraise:
                raise
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        return fallback(*args, **kw)

    def risk_penalty(self) -> float:
        return self.penalty if self.state is not CircuitState.CLOSED else 0.0


# ── HTTP client factory ──────────────────────────────────────────────────────
def http_client(timeout: float = 10.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, headers=get_settings().auth_headers())
