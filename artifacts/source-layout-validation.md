# Backend source-layout verification — October 26, 2026

The backend now uses `src/rfp/` with `main.py` as its FastAPI composition root. The import namespace stays `rfp`, and the three main facade paths remain unchanged. API views, REST schemas, middleware, A2A setup, configuration/telemetry and shared identifiers have explicit modules. Job-history SQL stays in `JobManager`, outside HTTP routes. All Python files are formatted consistently with Ruff.

## Executed checks

| Check | Actual result |
| --- | --- |
| Pre-refactor baseline | 19 tests; host OCR case skipped |
| Refactored host suite | 19 tests; host OCR case skipped |
| Refactored Docker suite | All 19 passed, including real Tesseract OCR, under 1 GiB and one CPU |
| Static checks | Ruff lint and formatting checks passed for source, tests, scripts and evaluation |
| Wheel build | Built the 0.1.0 wheel; all new modules and the CLI entry point present; no model/index data, tests or secrets packaged |
| Installed-package imports | Factory and facades import from `/usr/local/lib/python3.12/site-packages/rfp` with `python -I` and working directory `/tmp` |
| CLI/scripts | `python -I -m rfp --help`, load-test help and evaluation help passed |
| HTTP compatibility | All 17 pre-refactor HTTP paths and seven request/response schemas match; only documentation tags/summaries/descriptions ignored |
| Browser end-to-end | All 33 assertions passed against the rebuilt image and isolated copied data; six axe page/theme scans reported zero violations; no JavaScript runtime errors |
| Actual provider extraction | HTTP extraction of Bid2 completed: all 20 keys, 17 validated populated fields, three missing values, zero rejected fields |
| A2A | Existing discovery/RPC unit integration passed; live unauthorized discovery returned 401 and authenticated discovery returned 200 |
| Compose | Configuration validation passed |
| Local service update | Refactored image is serving the existing frontend on localhost |
| Data preservation | All three existing bid records are identical; all 12 existing index-file hashes are unchanged; no unavailable bids |
| Final frontend smoke | Lowercase demo login, service status, bid library and filtered real search passed through localhost:5173 |

The new application was exercised on port 8002 with a temporary frontend on 4174 before the local port-8000 service was updated. Test uploads and extraction writes used copied data. That staging service/frontend were removed after verification. The previous stopped container remains available as `bidwell-before-src-layout`; the active service retains the existing name `bidwell-demo-backend` and persistent mounts. Both `emplay-rfp-intelligence:src-layout` and `:local` point to the verified image.

## Evidence

- `layout-host-tests.txt`, `layout-container-tests.txt`: actual unittest output.
- `layout-installed-imports.txt`, `layout-package.json`: installed-package and wheel checks.
- `layout-http-contract.json`: comparison with the still-running old backend before replacement.
- `layout-browser-results.json`: actual browser assertions and accessibility findings.
- `layout-live-extraction.json`: actual Groq output from the rebuilt service.
- `layout-live-cutover.json`: final readiness, data-preservation and frontend checks.

This was an internal refactor. HTTP endpoints, environment variable names, index formats and the SQLite schema were preserved. No retrieval/LLM behavior or model choices were intentionally changed. Historical benchmarks and demo recordings remain historical; new Render/Vercel deployment was not performed. Passing these checks does not replace final human review of model-generated bid facts.
