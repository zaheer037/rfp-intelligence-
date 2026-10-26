"""Public liveness and readiness probes; no configuration or document disclosure."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def health_router() -> APIRouter:
    """Build process probes without loading inference models."""
    router = APIRouter(tags=["health"])

    @router.get("/health/live")
    async def live():
        """Liveness does not disclose secrets, documents or provider configuration."""
        return {"status": "alive"}

    @router.get("/health/ready")
    async def ready(request: Request):
        """Readiness indicates successful startup; model readiness is reported separately."""
        return JSONResponse(
            {"status": "ready" if request.app.state.ready else "starting"},
            status_code=200 if request.app.state.ready else 503,
        )

    return router
