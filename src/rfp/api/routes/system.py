"""Authenticated service status, live agent activity and Prometheus metrics."""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def system_router(settings, retrieval, service, jobs) -> APIRouter:
    """Expose operational state without secrets or local filesystem paths."""
    router = APIRouter(tags=["system"])

    @router.get("/status")
    async def status():
        """Report configured capabilities without exposing keys or local paths."""
        agents = service.activity.snapshot()
        return {
            "bids": retrieval.bid_ids(),
            "unavailable_bids": dict(retrieval.unavailable_bids),
            "llm_configured": bool(settings.groq_api_key.get_secret_value()),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "active_jobs": len(jobs.tasks),
            "agents": agents,
            "active_agents": sum(role["active_steps"] for role in agents),
        }

    @router.get("/metrics")
    async def metrics():
        """Expose Prometheus counters/histograms including process CPU/RSS metrics."""
        return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return router
