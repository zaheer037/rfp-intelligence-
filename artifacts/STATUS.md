# Executed verification — October 25, 2026

The backend and separate React frontend were exercised together with the configured Groq key. Keys are not included in artifacts. This is local end-to-end evidence; Render/Vercel deployment and final human acceptance remain outstanding.

## Executed checks

- Final Docker image: **22 tests passed**, including real Tesseract image/scanned-PDF OCR, with a 1024 MiB memory limit and one CPU. `container-tests.txt` contains the actual final output. Host tests pass with the OCR case skipped because Tesseract is provided by Docker.
- Backend checks cover incremental updates/deletions, corruption/snapshot recovery, input and document boundaries, authentication, upload limits, durable job admission/recovery, citation identity, strict citation-object parsing, bounded schema repair and the real Google ADK Runner with explicit test models. Injected test models are never selected by production configuration.
- Real ONNX embeddings/reranking and all nine supplied documents were indexed in Docker: Bid1 333 chunks; Bid2 34 chunks. Real HTTP search, status and Prometheus endpoints were exercised.
- Live Groq GPT-OSS 120B extraction, amended-deadline Q&A and warranty comparison were executed. Integration fixes handle ADK thought parts, Groq JSON mode, JSON-encoded citation objects, actionable bounded schema retries, trusted source-text binding and portal date/time evidence. Semantic validation still checks whether cited passages support each claim.
- The separate `rfp-web` frontend passes 33 real browser assertions, ten security/formatting/API-boundary tests and six automated accessibility page/theme scans. It covers all 20 fields, citations/documents, amendments, filtered search, live Q&A/comparison, asynchronous history/traces, metrics/evaluation and a genuinely unseen HTML upload. Its synthetic upload was removed after verification.
- 18 manually checked retrieval labels were evaluated across three configurations with the real BGE-small and TinyBERT models. This is a small development evaluation, not a held-out accuracy benchmark.
- The original 1 GiB / one CPU search baseline returned 100/100 HTTP 200 responses, 2.314 successful requests/s, p95 1.014 s and p99 1.260 s. A separate concurrency-four overload run returned 11 successes and 89 controlled 429 responses, zero server errors. Highest captured cgroup memory peak was 492.9 MiB, with zero OOM counters. These measurements describe that local workload and corpus, not arbitrary corpus sizes or Render latency. The load benchmark was not rerun after the additional workspace routes/provider fixes.
- Docker build and Compose validation passed. Render blueprint and Vercel configuration have schema-validation evidence. The backend GitHub Actions workflow is restored per request; remote runs are tracked in the repository Actions tab.

## Deliverables and practical limits

`Bid1.json` and `Bid2.json` are actual provider outputs, not hand-filled examples. Both final API runs completed with zero rejected fields: Bid1 has 15 populated validated fields and five missing-evidence fields; Bid2 has 17 and three. Each record contains all twenty keys and reports its own validation, missing evidence and errors. A completed status means the workflow's checks completed; it does not mean every fact was found or independently human-approved. Repeated generations can produce different omissions. Preserve and inspect partial runs rather than treating them as accepted.

`qa-log.json` contains twelve actual answers: ten include source citations and two explicitly abstain. The two abstentions concern questions whose labeled evidence exists, so they are false negatives, not proof that the source documents lack that information. Do not report this as perfect answer accuracy. The extra two cited questions cover the requested laptop model and display resolution. Retrieval metrics are measured separately and do not imply equivalent generated-answer accuracy.

`sample-trace.jsonl` is an actual full successful extraction trace, including inputs, retrieved evidence, model outputs, token counts and validation/repair steps. Content capture is opt-in; normal deployment traces redact content. Restrict access when processing non-public documents.

The frontend's QA report includes mobile Lighthouse, responsive screenshots and deployment instructions. The demo is shared and public, with server-side backend keys; it is not a private multi-tenant identity system. Configure a shared edge rate limit before exposing paid provider usage.

## Remaining submission/deployment work

1. Human-review the final twenty fields for each bid against the supplied documents, including nulls and the Bid1 portal/addendum timezone discrepancy. Review the two answer abstentions and any partial extraction runs.
2. Deploy the Docker backend to Render with persistent storage and the separate React repository to Vercel; configure real environment values, then repeat HTTPS/session/source-download/async-job smoke checks on those URLs.
3. Provide repository access and review the included final narrated 8m19s demo in ../demo/README.md.

The supplied Render blueprint currently uses its documented `1c-2g` plan (2 GB); the application was independently tested at the user's requested 1 GB limit. Select the available account-specific plan and ensure the persistent mount is writable by UID 10001. No Render or Vercel deployment was performed.

Chat-first revision: `/status` now exposes seven role descriptions and actual active execution counts, including concurrent extraction specialists. Counts reset on step completion, failure or cancellation; idle is zero. Login opens chat and the other tools remain in the sidebar.

## Source-layout refactor — October 26, 2026

Moved application code to `src/rfp/`, added the explicit `main.py` factory, separated routes/middleware/core/shared helpers, and updated packaging, Docker and CI. All 19 container tests and 33 browser assertions pass; a real HTTP extraction completed with 17 validated fields, three missing values and zero rejected. The working local service was updated after staged verification. Existing bid records and all index-file bytes were preserved. See [source-layout-validation.md](source-layout-validation.md) for executed checks and evidence.

## Minimal environment and Docker Hub release — October 26, 2026

Reduced the backend .env from 20 variables to the two required credentials without changing effective local defaults. The image now enables production key validation and derives its A2A URL from Railway-provided domain metadata. All 22 container tests pass under 1 GiB/one CPU. A fresh root-owned-volume simulation with PORT=8080 completed first-use model downloads, Bid2 indexing (34 chunks, no warnings), real filtered search and cited Groq Q&A, then retained indexes/jobs/model caches after restart. The tested container digest is recorded in dockerhub-release.json; registry identifiers are redacted in the public report. See ../RAILWAY.md for platform setup and first indexing. No Railway deployment was performed.
