"""Single entry point for the server, standalone search and bid-agent workflows."""

import argparse
import asyncio
import json
import os

from rfp.core.config import Settings
from rfp.core.observability import configure_logging


def main():
    """Run API or pipeline commands with the same configuration and implementations."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    index = sub.add_parser("index")
    index.add_argument("folder")
    index.add_argument("--bid-id")
    run = sub.add_parser("run", help="Index a folder and extract the complete bid record")
    run.add_argument("folder")
    run.add_argument("--bid-id")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--bid-id")
    search.add_argument("--mode", choices=["dense", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    extract = sub.add_parser("extract")
    extract.add_argument("bid_id")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--bid-id", action="append")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "rfp.main:create_app",
            factory=True,
            host="0.0.0.0",
            port=args.port,
            workers=1,
            limit_concurrency=64,
            timeout_keep_alive=5,
            timeout_graceful_shutdown=1800,
            access_log=False,
        )
        return
    from rfp.agents.agent import Agent
    from rfp.ingestion.indexing import Indexing
    from rfp.search.models import LocalEmbeddings, LocalReranker
    from rfp.search.retrieval import Retrieval

    settings = Settings()
    configure_logging(settings.log_level)
    indexing = Indexing(settings, LocalEmbeddings(settings))
    retrieval = Retrieval(indexing, indexing.embeddings, LocalReranker(settings))
    if args.command not in {"index", "run"}:
        retrieval.reload()
    agent = Agent(settings, indexing, retrieval)
    if args.command == "index":
        output = asyncio.run(agent.ingest(args.folder, args.bid_id))
    elif args.command == "run":
        output = asyncio.run(agent.process_folder(args.folder, args.bid_id)).model_dump()
    elif args.command == "search":
        output = [h.model_dump() for h in retrieval.search(args.query, bid_id=args.bid_id, mode=args.mode)]
    elif args.command == "extract":
        output = asyncio.run(agent.extract(args.bid_id)).model_dump()
    else:
        output = asyncio.run(agent.ask(args.question, args.bid_id)).model_dump()
    print(json.dumps(output, indent=2))
