"""Shared ASGI assembly for MCP servers: streamable-HTTP transport + /health + /metrics + bearer auth."""
from __future__ import annotations

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from p2p.common.ops import BearerAuthMiddleware, configure_logging, metrics_endpoint


def build_asgi(mcp: FastMCP, service: str):
    configure_logging(service)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": service})

    mcp.custom_route("/metrics", methods=["GET"])(metrics_endpoint)
    return BearerAuthMiddleware(mcp.http_app(path="/mcp"))
