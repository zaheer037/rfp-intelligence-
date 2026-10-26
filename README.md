# RFP Intelligence — Agentic RAG for bid documents

**[▶ Watch the narrated demo — 8m 19s](https://drive.google.com/file/d/1gLtL16zGsFf8m7f_c9N-Hp1YCBAFMel1/view)** · [Download the video](demo/Bidwell_Demo_with_voiceover.mp4)

A FastAPI backend that ingests HTML, PDFs and images, retrieves cited evidence with FAISS/BM25, and uses Google ADK agents with Groq GPT-OSS 120B to extract bid requirements and answer questions. Local quantized ONNX models keep embedding and reranking small enough for a tested **1 GB RAM / one CPU** deployment.

[Deliverables checklist](DELIVERABLES.md) · [Architecture diagram](ARCHITECTURE.md) · [Verification and limitations](artifacts/STATUS.md) · [Railway deployment](RAILWAY.md)

## Assignment outputs

| Deliverable | Files |
| --- | --- |
| Twenty-field extraction with values, citations and confidence | [Bid1.json](artifacts/Bid1.json), [Bid2.json](artifacts/Bid2.json) |
| Retrieval evaluation | [18 labeled questions](eval/questions.json), [measured report](artifacts/retrieval-evaluation.json) |
| Sample Q&A | [12 actual answers](artifacts/qa-log.json): ten cited answers and two abstentions |
| Full extraction trace | [sample-trace.jsonl](artifacts/sample-trace.jsonl) |
| Agent/search architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Narrated demo | [Video and metadata](demo/README.md) |

The supplied input documents are in `bids/Bid1/` and `bids/Bid2/`. The React UI is maintained in a separate repository; this repository contains its demonstration video but no frontend source.

## Setup

Use Python 3.12 and install Tesseract for OCR, or use Docker, which includes it.

```bash
git clone https://github.com/zaheer037/rfp-intelligence-.git
cd rfp-intelligence-
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
pip install -e '.[dev]'
cp .env.example .env
```

Set the two credentials in `.env`: `GROQ_API_KEY` and a strong `RFP_API_KEY` of at least 24 characters. No API keys are committed. Optional settings are defined in `src/rfp/core/config.py` and use the `RFP_` prefix. `.env` is loaded from the current working directory; run the commands below from the repository root.

The default input root is `./bids`, runtime storage is `./storage`, and model cache is `./models`. Add new folders under `bids/`, or override `RFP_INPUT_ROOT`. External links within documents are not fetched. Models download on first use; readiness does not imply warmed models or a reachable Groq service.

## Run each mode

```bash
# Start the authenticated API on 0.0.0.0:8000 (or Railway's PORT).
rfp serve
# Equivalent: python -m rfp serve
# ASGI factory: rfp.main:create_app

# Index both supplied folders, without invoking the language model.
rfp index Bid1
rfp index Bid2

# Standalone retrieval: dense, hybrid or hybrid_rerank.
rfp search 'final submission deadline' --bid-id Bid1 --mode hybrid_rerank

# Multiagent twenty-field extraction.
rfp extract Bid1 > artifacts/Bid1.json
rfp extract Bid2 > artifacts/Bid2.json

# Cited questions and cross-bid comparison.
rfp ask 'What laptop model is requested?' --bid-id Bid2
rfp ask 'Compare the warranty requirements of both bids.'

# Ingest an additional bids/NewBidFolder and extract in one command.
rfp run NewBidFolder --bid-id NewBid > artifacts/NewBid.json
```

Extraction/Q&A require the configured Groq key. Search, ingestion and evaluation use local CPU models. Generated results can vary; preserve the supplied artifacts before overwriting them with another run.

## API

All routes except `/health/live` and `/health/ready` require `X-API-Key`, including `/docs`, `/openapi.json`, `/metrics` and A2A discovery/RPC. Use HTTPS when deployed.

```bash
# Export the same RFP_API_KEY configured in .env before these examples.
curl -H "X-API-Key: $RFP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"folder":"Bid1"}' http://localhost:8000/index

curl -H "X-API-Key: $RFP_API_KEY" \
  'http://localhost:8000/search?q=deadline&bid_id=Bid1&top_k=5&mode=hybrid_rerank'

curl -H "X-API-Key: $RFP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"bid_id":"Bid1"}' http://localhost:8000/extract

curl -H "X-API-Key: $RFP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Compare warranty requirements","bid_ids":["Bid1","Bid2"]}' \
  http://localhost:8000/ask
```

`POST /index`, `/extract` and `/ask` return **202** with a `job_id`. Poll `GET /jobs/{job_id}` until `completed`, `partial` or `failed`. SQLite retains job history; interrupted jobs become failed on restart. At capacity the API returns **429** with `Retry-After`; missing Groq configuration rejects LLM jobs before admission. Additional workspace routes expose uploaded documents, bid records and traces to the separate frontend.

`GET /search` accepts `bid_id`, `doc_type`, `addendum_number`, `top_k` and ranking `mode`. Results include passages, source metadata and scores. PDF citations use physical 1-based pages; HTML citations have `page=null` and a section locator. `/status` lists indexed bids, unavailable snapshots, seven agent roles and actual current `active_agents`/`active_steps`. Idle counts are zero. Agent status is process-local and shared across clients.

## Code responsibilities

```text
src/rfp/
├── main.py                 # FastAPI composition, startup and shutdown
├── cli.py                  # API and independent pipeline commands
├── __main__.py             # python -m rfp
├── api/
│   ├── routes/             # Health, operations, research and workspace views
│   ├── middleware.py       # API authentication, limits and HTTP metrics
│   ├── schemas.py          # HTTP request contracts
│   ├── jobs.py             # Durable bounded job registry
│   └── a2a.py              # Google ADK A2A adapter
├── ingestion/indexing.py   # Indexing: parse, OCR, chunk, persist, update
├── search/
│   ├── retrieval.py        # Retrieval: filter, search, fuse, rerank, cite
│   └── models.py           # Lazy local embedding and reranker inference
├── agents/
│   ├── agent.py            # Agent facade, ADK orchestration and run state
│   ├── prompts.py          # Roles, guardrails and extraction field groups
│   └── validation.py       # Deterministic citation and format validation
├── core/                   # Environment settings, logs, traces and metrics
├── utils/                  # Shared identifier validation
└── schemas.py              # Typed domain and agent messages
```

The API delegates to the same `Indexing`, `Retrieval` and `Agent` classes as the CLI. Models load lazily; imports/startup do not load weights. Tests, evaluation, load testing and deliverable collection live in `tests/`, `eval/` and `scripts/`.

## Design decisions

**Chunking and ingestion.** Heading/page-aware chunks target 220 words with 35-word overlap to fit the embedding model's 512-token context. Long table rows split with repeated column headers. PyMuPDF extracts native PDF text, pdfplumber handles ruled tables, BeautifulSoup handles HTML, and bounded Tesseract OCR handles scans and image regions. Each chunk records bid, document type, addendum number, filename, physical page/section, method and content hash. Decorative images may be ignored; numeric OCR output needs review.

**Embeddings and storage.** Quantized ONNX BGE-small (`BAAI/bge-small-en-v1.5`, approximately 67 MB weights, 384 dimensions) runs locally through FastEmbed without PyTorch. This replaces the initial large Mistral proposal to fit the small-host requirement. Normalized vectors use FAISS exact inner-product search; SQLite stores chunk metadata/text. BM25 reconstructs from persisted text. Per-file/chunk hashes avoid unchanged embeddings; changed/deleted files replace old content. Atomic snapshot publication protects consistency. Model/parser/chunking fingerprints reject incompatible snapshots until rebuilt. Unreadable replacements preserve prior content with warnings.

**Retrieval.** Query expansion preserves the original terms and product identifiers. Metadata constraints filter the eligible corpus before ranking. Weighted reciprocal rank fusion (constant 60; dense:keyword 2:1) combines FAISS and BM25. A local FlashRank TinyBERT cross-encoder (approximately 4 MB weights) reranks candidates; final rank blends prior rank 75% and cross-encoder rank 25% through RRF. These configurable weights were developed on this small dataset, not established as universally optimal. Filtered subset construction is O(n).

**Agent framework and communication.** Google ADK `LlmAgent`/`Runner`, with LiteLLM's Groq adapter, runs GPT-OSS 120B. A planner, three concurrent extraction specialists, addendum reconciler, semantic critic and Q&A/report role exchange Pydantic-validated messages through per-run typed state. Ingestion/retrieval are deterministic tools. An ADK A2A root exposes the same facade; no separate remote service per agent is required. The Python orchestrator controls the workflow and bounds retries; the planner does not execute arbitrary commands. See the [communication diagram](ARCHITECTURE.md).

**Prompts and evidence.** `agents/prompts.py` defines role instructions, field groups and cited-or-abstain guardrails. Retrieved documents are evidence rather than instructions; specialists receive relevant passages, not whole documents. Extraction uses structured responses separately from function calling. The evidence registry binds returned citations to actual retrieved chunks. Deterministic checks and an LLM critic validate support and formats; targeted repair runs at most twice. Addendum changes require field-specific before/after evidence. Ambiguous timezones remain flagged instead of silently converted. Confidence is a heuristic, not a calibrated probability. Blank affidavits do not prove identities or signatures.

## Evaluation and verification

```bash
python -m unittest discover -s tests -v
ruff check src tests scripts eval
ruff format --check src tests scripts eval

# Real local-model evaluation, after indexing both bids.
python eval/evaluate.py

# Regenerate live provider outputs; detailed trace capture is opt-in.
RFP_TRACE_CONTENT=true python scripts/collect_deliverables.py

# Warm-search load test against an already running API.
python scripts/load_test.py --requests 100 --concurrency 2
```

`eval/questions.json` contains 18 manually checked question/source-passage labels across both bids. Recall@5 and MRR@5 match filename, physical page and expected passage text. The committed measured configurations are:

| Configuration | Recall@5 | MRR@5 |
| --- | --- | --- |
| BGE-small dense | 1.000 | 0.773 |
| BGE-small + BM25 | 0.944 | 0.780 |
| BGE-small + BM25 + TinyBERT rank blend | 0.944 | 0.807 |

This is a development set, not a held-out benchmark. Retrieval scores do not imply equivalent answer/extraction accuracy. The supplied final records have 15 populated fields for Bid1 and 17 for Bid2; all twenty keys exist, with missing evidence represented explicitly. Ten Q&A entries contain citations; two abstain despite available source evidence and are recorded false negatives. Human review remains necessary, especially for the Bid1 portal/addendum timezone discrepancy.

**22 container tests passed**, including real scanned-PDF/image OCR, under 1 GB / one CPU. Unit/integration tests explicitly inject model doubles while exercising the actual ADK Runner; live ONNX/Groq evidence is recorded separately. The original warm-search load run completed 100/100 HTTP 200 responses, p95 1.014 s, p99 1.260 s and 2.314 successful requests/s. A concurrency-four overload run recorded 11 successes and 89 explicit 429s, no server errors. Captured memory peak was 492.9 MiB with no OOM events. These are historical local measurements for this corpus; they are not a deployed-service capacity guarantee or a new benchmark after every source change.

The commands above run tests, lint/format checks and retrieval evaluation locally. Docker supplies Tesseract for OCR testing. Automated GitHub Actions workflows are omitted from this submission. See [STATUS.md](artifacts/STATUS.md) for executed checks and remaining limitations.

## Logging and production boundaries

Structured UTC JSON logs include request/run IDs, stage, status and latency. Prometheus measures HTTP/stage/search/model latency, failures, retries, actual provider token usage, field validation and process CPU/RSS/GC. Query text, arbitrary filenames and bid IDs are not metric labels. Monetary costs are not estimated from potentially stale prices.

Trace content is redacted by default. Opt-in detailed traces and stored results may contain document content; restrict disk access and manage retention. The public sample trace uses the supplied assignment documents. The API bounds request bodies, jobs, concurrent searches/model calls, query length, files/pages/pixels, OCR/model timeouts and retries. Provider failures return explicit failed/partial states. Graceful shutdown drains jobs; forced interruptions remain visible on restart.

Use **one process, one replica and persistent storage**. Local FAISS snapshots/SQLite are not a shared horizontally scalable store. Larger or untrusted workloads need isolated ingestion workers and shared persistence. Semantic answer caching, configurable go/no-go scoring and monetary cost accounting are not implemented. Cited cross-bid comparison is supported through Q&A; a dedicated side-by-side schema is not provided.

## Docker and deployment

```bash
# Build from this repository root; bids are included, secrets/models are excluded.
docker build -t rfp-intelligence .

# Reads the two credentials from .env; mounts persistent data.
docker compose up --build
```

Docker includes Tesseract and CPU ONNX Runtime, runs as UID 10001 by default, enables production key validation, listens on `$PORT`, and stores runtime data and models under `/var/data`. First inference downloads weights. A new volume must index Bid1 and Bid2 before search/extraction. Keep the volume writable by the runtime UID.

The published Docker Hub release is [`[registry-redacted]/rfp-intelligence:2026-10-26-railway`](https://hub.docker.com/r/[registry-redacted]/rfp-intelligence). Its tested digest is in `artifacts/dockerhub-release.json`. Follow [RAILWAY.md](RAILWAY.md) for Railway's volume UID setting, credentials, readiness probe and first indexing. Railway deployment has not been performed. `render.yaml` is an alternative Blueprint; its selected `1c-2g` plan has 2 GB, while local validation independently used a 1 GB container limit. The React frontend is deployed separately to Vercel with server-side backend credentials.

## References

[Google ADK](https://google.github.io/adk-docs/) · [FAISS](https://faiss.ai/) · [FastEmbed](https://qdrant.github.io/fastembed/) · [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) · [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
