"""FastAPI composition root: build dependencies, routes and application lifecycle."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rfp.agents.agent import Agent
from rfp.api.a2a import create_a2a_app
from rfp.api.jobs import JobManager
from rfp.api.middleware import install_middleware
from rfp.api.routes.health import health_router
from rfp.api.routes.research import research_router
from rfp.api.routes.system import system_router
from rfp.api.routes.workspace import workspace_router
from rfp.core.config import Settings
from rfp.core.observability import configure_logging
from rfp.ingestion.indexing import Indexing
from rfp.search.models import LocalEmbeddings, LocalReranker
from rfp.search.retrieval import Retrieval


def create_app(settings: Settings | None = None, *, embeddings=None, reranker=None, agent_factory=Agent):
    """Construct one service instance, with injectable model boundaries for tests."""
    settings = settings or Settings()
    configure_logging(settings.log_level)
    indexing = Indexing(settings, embeddings or LocalEmbeddings(settings))
    retrieval = Retrieval(indexing, indexing.embeddings, reranker or LocalReranker(settings))
    service = agent_factory(settings, indexing, retrieval)
    jobs = JobManager(settings)
    a2a_app = create_a2a_app(settings, service) if agent_factory is Agent else None

    @asynccontextmanager
    async def lifespan(app):
        """Load metadata, manage the A2A lifespan and drain accepted jobs on shutdown."""
        await asyncio.to_thread(retrieval.reload)
        app.state.ready = True
        try:
            if a2a_app is not None:
                async with a2a_app.router.lifespan_context(a2a_app):
                    yield
            else:
                yield
        finally:
            app.state.ready = False
            await jobs.close()

    app = FastAPI(title="RFP Intelligence", version="0.1.0", lifespan=lifespan)
    app.state.ready = False
    app.state.service, app.state.retrieval, app.state.jobs = service, retrieval, jobs
    install_middleware(app, settings)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        """Translate rejected domain inputs to the established HTTP error contract."""
        return JSONResponse({"detail": str(exc)}, status_code=400)

    app.include_router(health_router())
    app.include_router(system_router(settings, retrieval, service, jobs))
    app.include_router(research_router(settings, indexing, retrieval, service, jobs))
    app.include_router(workspace_router(settings, indexing, retrieval, service, jobs))
    if a2a_app is not None:
        app.mount("/a2a", a2a_app)
    return app
