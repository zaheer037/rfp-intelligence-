# Assignment deliverables

This repository contains the backend and submission evidence. The React frontend source will live in its own repository. Artifacts are actual recorded outputs; tests using injected models are distinguished from live provider checks in [STATUS.md](artifacts/STATUS.md).

| Required deliverable | Location | Evidence and limits |
| --- | --- | --- |
| Ingestion pipeline | `src/rfp/ingestion/` | HTML, PDF, tables, image/scanned-PDF OCR, incremental snapshots |
| Standalone RAG search engine | `src/rfp/search/` | Local BGE-small, FAISS/BM25 fusion, TinyBERT rank blend, metadata filters |
| Multiagent system | `src/rfp/agents/` | Google ADK/Groq planner, concurrent specialists, addendum reconciliation, critic and Q&A; typed state and bounded repairs |
| API and CLI | `src/rfp/api/`, `src/rfp/cli.py`, `src/rfp/main.py` | Authenticated async FastAPI, ADK A2A adapter, `rfp` commands |
| Setup, dependencies, modes, design decisions and prompts | [README.md](README.md), `requirements.lock`, `src/rfp/agents/prompts.py` | Python 3.12 instructions; indexing/search/extraction/Q&A/full-run examples |
| Architecture and agent communication | [ARCHITECTURE.md](ARCHITECTURE.md) | Implemented Mermaid diagram and message/state explanation |
| JSON per bid folder | [Bid1.json](artifacts/Bid1.json), [Bid2.json](artifacts/Bid2.json) | All 20 keys, values, source citations and confidence; explicit nulls for missing evidence |
| Retrieval evaluation questions and report | [questions.json](eval/questions.json), [retrieval-evaluation.json](artifacts/retrieval-evaluation.json) | 18 development questions; measured Recall@5 / MRR@5 for dense, hybrid and hybrid+rerank |
| At least ten cited sample Q&A answers | [qa-log.json](artifacts/qa-log.json) | 12 recorded answers: 10 cited answers, 2 explicit abstentions/false negatives |
| Full extraction agent trace | [sample-trace.jsonl](artifacts/sample-trace.jsonl) | Successful actual extraction, retrieved inputs, model outputs, usage and validation/repair steps |
| Demo video, 5–10 minutes | [Narrated demo](demo/README.md) | 8m19s, 1080p with audio; local MP4 and Google Drive link |

## Optional bonuses

| Bonus | Status |
| --- | --- |
| React search/chat UI | Implemented separately; source intentionally excluded here. Shown in the demo |
| Scanned-PDF and image/table OCR | Implemented; real Tesseract container test |
| Bid comparison | Cited cross-bid Q&A/report supported. No dedicated structured side-by-side report schema |
| Configurable company capabilities and go/no-go | Not implemented |
| Semantic caching and cost/latency tracking | Latency and actual token metrics implemented. No semantic answer cache or monetary cost accounting; incremental indexing/model caching is different |
| Docker and CI tests/evaluation | Dockerfile and Compose included; local tests/evaluation executed. Automated CI omitted per request |

## Before submission

Human-review the populated values, nulls and source citations, particularly the Bid1 timezone discrepancy. Extraction completion means the workflow checks finished, not that every required fact was found. The included final recording has narration. Deploy the backend using [RAILWAY.md](RAILWAY.md), then smoke-test the deployed HTTPS URL. Deployment is not implied by the local evidence.
