"""Bounded HTTP load test; writes actual latency/error/throughput measurements as JSON."""

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path

import httpx

from rfp.core.config import Settings


async def run(args):
    """Measure successful requests separately from admission rejections and server errors."""
    statuses, latencies, success_latencies = Counter(), [], []
    start = time.perf_counter()
    next_request = 0
    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=args.timeout,
        headers={"X-API-Key": Settings().api_key.get_secret_value()},
        trust_env=False,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def worker():
            nonlocal next_request
            while next_request < args.requests:
                next_request += 1
                began = time.perf_counter()
                try:
                    response = await client.get(
                        args.path,
                        params={"q": args.query, "bid_id": args.bid_id, "mode": args.mode}
                        if args.path == "/search"
                        else {},
                    )
                    outcome = str(response.status_code)
                    if response.is_success and args.path == "/search":
                        body = response.json()
                        if not body.get("hits") or not all(
                            h.get("citation", {}).get("chunk_id") for h in body["hits"]
                        ):
                            outcome = "invalid_search_result"
                    statuses[outcome] += 1
                    if outcome == "200":
                        success_latencies.append(time.perf_counter() - began)
                except httpx.HTTPError:
                    statuses["transport_error"] += 1
                latencies.append(time.perf_counter() - began)

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    duration = time.perf_counter() - start

    def percentile(values, p):
        return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None

    result = {
        "url": args.url,
        "path": args.path,
        "mode": args.mode,
        "requests": len(latencies),
        "concurrency": args.concurrency,
        "duration_seconds": duration,
        "requests_per_second": len(latencies) / duration,
        "statuses": dict(statuses),
        "successes_per_second": len(success_latencies) / duration,
        "successful_latency_seconds": {
            "p50": percentile(success_latencies, 0.5),
            "p95": percentile(success_latencies, 0.95),
            "p99": percentile(success_latencies, 0.99),
        },
        "scope": args.scope,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    unexpected = sum(v for k, v in statuses.items() if k != "200")
    return 1 if unexpected / max(1, len(latencies)) > args.max_error_rate else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default="/search", choices=["/search", "/health/live"])
    parser.add_argument("--query", default="What is the submission deadline?")
    parser.add_argument("--bid-id", default="Bid1")
    parser.add_argument("--mode", default="hybrid_rerank", choices=["dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--scope", default="live local embedding and reranker models")
    parser.add_argument("--output", default="artifacts/load-test.json")
    options = parser.parse_args()
    if not 1 <= options.concurrency <= 100 or not 1 <= options.requests <= 100000:
        parser.error("concurrency must be 1–100 and requests 1–100000")
    raise SystemExit(asyncio.run(run(options)))
