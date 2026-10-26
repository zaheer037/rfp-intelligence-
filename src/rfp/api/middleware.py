"""HTTP authentication, request limits, correlation IDs and transport metrics."""

import asyncio
import hmac
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rfp.core.config import Settings
from rfp.core.observability import HTTP_LATENCY, HTTP_REQUESTS, REQUEST_ID, event


class BodyLimit:
    """Enforce request size even with chunked transfer or a forged Content-Length."""

    def __init__(self, app, limit=32_768):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0
        limit = 3_000_000 if scope.get("path") == "/upload" else self.limit
        messages = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > limit:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay, send)


def install_middleware(app: FastAPI, settings: Settings) -> None:
    """Apply the same protection and measurement to REST and mounted A2A routes."""
    request_slots = asyncio.Semaphore(32)
    app.add_middleware(BodyLimit)

    @app.middleware("http")
    async def protect_and_measure(request: Request, call_next):
        """Authenticate all data/metrics/docs routes and attach a server-generated request ID."""
        request_id = uuid.uuid4().hex
        token = REQUEST_ID.set(request_id)
        start, status = time.perf_counter(), 500
        try:
            if request.url.path not in {"/health/live", "/health/ready"}:
                expected = settings.api_key.get_secret_value()
                supplied = request.headers.get("x-api-key", "")
                if not expected or not hmac.compare_digest(expected, supplied):
                    status = 401
                    return JSONResponse({"detail": "Valid X-API-Key required"}, status_code=status)
            try:
                await asyncio.wait_for(request_slots.acquire(), timeout=0.01)
            except TimeoutError:
                status = 429
                return JSONResponse({"detail": "Server busy"}, status_code=429, headers={"Retry-After": "2"})
            try:
                response = await call_next(request)
            finally:
                request_slots.release()
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        except Exception as exc:
            event("http_failed", error_type=type(exc).__name__)
            return JSONResponse({"detail": "Internal error", "request_id": request_id}, status_code=500)
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if request.url.path.startswith("/a2a"):
                route = "/a2a"
            latency = time.perf_counter() - start
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            HTTP_LATENCY.labels(request.method, route).observe(latency)
            event("http_request", method=request.method, route=route, status=status, latency_seconds=latency)
            REQUEST_ID.reset(token)
