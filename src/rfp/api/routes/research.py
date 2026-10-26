"""Indexing, independent retrieval, extraction, Q&A and durable job polling."""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from rfp.api.schemas import AskRequest, ExtractRequest, IndexRequest
from rfp.schemas import DocType


def research_router(settings, indexing, retrieval, service, jobs) -> APIRouter:
    """Adapt HTTP inputs to the shared pipeline while bounding search and job admission."""
    router = APIRouter(tags=["research"])
    search_slots = asyncio.Semaphore(settings.max_search_concurrency)

    def require_llm():
        if not settings.groq_api_key.get_secret_value():
            raise HTTPException(503, "Configure GROQ_API_KEY before running agent tasks")

    def submit(kind, operation):
        try:
            identifier = jobs.submit(kind, operation)
        except OverflowError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
        return {"job_id": identifier, "status_url": f"/jobs/{identifier}"}

    @router.post("/index", status_code=202)
    async def index(body: IndexRequest):
        """Accept trusted local folders; processing is a durable bounded background job."""
        indexing.resolve_folder(body.folder)
        return submit("index", lambda _: service.ingest(body.folder, body.bid_id))

    @router.get("/search")
    async def search(
        q: str = Query(min_length=1, max_length=2000),
        bid_id: str | None = None,
        doc_type: DocType | None = None,
        addendum_number: int | None = Query(None, ge=1),
        top_k: int = Query(5, ge=1, le=50),
        mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    ):
        """Search indexed bids with metadata restrictions and traceable passages."""
        try:
            await asyncio.wait_for(search_slots.acquire(), timeout=0.05)
        except TimeoutError as exc:
            raise HTTPException(429, "Search capacity reached", headers={"Retry-After": "2"}) from exc
        try:
            hits = await asyncio.to_thread(
                retrieval.search,
                q,
                bid_id=bid_id,
                doc_type=doc_type,
                addendum_number=addendum_number,
                top_k=top_k,
                mode=mode,
            )
        except (ImportError, OSError) as exc:
            raise HTTPException(503, "Local models unavailable; inspect server setup") from exc
        finally:
            search_slots.release()
        return {"query": q, "hits": [h.model_dump() | {"citation": h.citation().model_dump()} for h in hits]}

    @router.post("/extract", status_code=202)
    async def extract(body: ExtractRequest):
        """Schedule complete 20-field extraction for one indexed bid."""
        require_llm()
        if body.bid_id not in retrieval.bid_ids():
            raise HTTPException(404, "Bid is not indexed")
        return submit("extract", lambda run_id: service.extract(body.bid_id, run_id))

    @router.post("/ask", status_code=202)
    async def ask(body: AskRequest):
        """Schedule cited Q&A or a cross-bid comparison without holding HTTP open."""
        require_llm()
        ids = body.bid_ids or retrieval.bid_ids()
        if not ids or any(bid not in retrieval.bid_ids() for bid in ids):
            raise HTTPException(404, "Requested bids must be indexed")
        return submit("ask", lambda run_id: service.ask(body.question, ids, run_id))

    @router.get("/jobs/{job_id}")
    async def job(job_id: str):
        """Poll durable status and retrieve successful/partial outputs."""
        if not (record := jobs.get(job_id)):
            raise HTTPException(404, "Job not found")
        return record

    return router
