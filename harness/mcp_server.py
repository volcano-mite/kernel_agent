"""Expose one run's public tools over authenticated Streamable HTTP."""

import asyncio
import re
from typing import Any
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from .evaluator import MAX_SOURCE_BYTES, RunError


MAX_BODY_BYTES = 6 * MAX_SOURCE_BYTES + 8192


def create_app(evaluator, run_id):
    store = evaluator.store
    config = store.config(run_id)
    server = FastMCP(
        "Kernel Agent",
        stateless_http=True,
        json_response=True,
        max_request_body_size=MAX_BODY_BYTES,
        transport_security=TransportSecuritySettings(
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "host.containers.internal:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                             "http://host.containers.internal:*"],
        ),
    )

    async def invoke(requested_run, function, *args):
        if requested_run != run_id:
            raise ValueError("Tool is scoped to the authenticated run")
        try:
            return await asyncio.to_thread(function, run_id, *args)
        except RunError as error:
            raise ValueError(str(error)) from error
        except FileNotFoundError as error:
            raise ValueError("Unknown candidate for this run") from error

    @server.tool()
    async def evaluate_kernel(run_id: str, source: str) -> dict[str, Any]:
        """Archive and evaluate the complete contents of rmsnorm.cu on the public cases."""
        return await invoke(run_id, evaluator.evaluate, source)

    if config["variant"] in {"D", "F"}:
        @server.tool()
        async def sanitize_kernel(run_id: str, candidate_id: str) -> dict[str, Any]:
            """Run Compute Sanitizer on an archived candidate and a public case."""
            return await invoke(run_id, evaluator.diagnostic, candidate_id, "sanitize")

        @server.tool()
        async def profile_kernel(run_id: str, candidate_id: str) -> dict[str, Any]:
            """Collect Nsight Compute counters for an archived candidate and a public case."""
            return await invoke(run_id, evaluator.diagnostic, candidate_id, "profile")

    app = server.streamable_http_app()
    app.add_middleware(RunAuthorization, store=store, run_id=run_id)
    return app


class RunAuthorization:
    def __init__(self, app, store, run_id):
        self.app = app
        self.store = store
        self.run_id = run_id

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        authorization = Request(scope).headers.get("authorization", "")
        match = re.fullmatch(r"Bearer ([A-Za-z0-9_-]{43})", authorization, flags=re.IGNORECASE)
        if not match or not self.store.authenticate(self.run_id, match[1]):
            response = JSONResponse({"error": "Run authentication required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
