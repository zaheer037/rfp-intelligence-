# Deploy the published backend image on Railway

Use `[registry-redacted]/rfp-intelligence:2026-10-26-railway` (fixed release) or `:latest` in a Railway **Docker Image** service. The image is Linux amd64 and includes the backend, Tesseract and the supplied public Bid1/Bid2 source documents. Application keys, local indexes, jobs, model caches and frontend files are not baked into it.

## Service setup

1. Attach one persistent volume at **`/var/data`**. It holds `storage/` (FAISS, SQLite, uploads and traces) and `models/` (downloaded local ONNX weights).
2. Add the two application variables:

   ```dotenv
   RFP_API_KEY=<your existing backend key, at least 24 characters>
   GROQ_API_KEY=<your Groq key>
   ```

3. Set **`RAILWAY_RUN_UID=0`** for the root-owned Railway volume. This image otherwise defaults to UID 10001. This is a Railway volume-permissions setting, not another application configuration knob. See Railway's [volume permissions guidance](https://docs.railway.com/volumes/reference).
4. Set the healthcheck path to **`/health/ready`**. Keep the image's default start command. It listens on `0.0.0.0:$PORT`; Railway supplies `PORT`. Generate a public domain. See [Railway healthchecks](https://docs.railway.com/deployments/healthchecks).
5. Use **one replica**, with at least the tested **1 GB RAM / one CPU** budget. This is a local FAISS/SQLite service with one writer, not a horizontally shared database.

The image already supplies production mode, absolute storage/input/model paths and CPU limits for inference. The language model, embedding/reranker names, chunking and admission limits have defaults in `src/rfp/core/config.py`. You do not need to copy those defaults into Railway variables. A2A discovery uses Railway's automatically supplied `RAILWAY_PUBLIC_DOMAIN`; `RFP_PUBLIC_BASE_URL` is an optional override for another deployment/domain.

## First deployment

A new volume starts without indexes or model weights. Readiness means the API has started, not that indexing is complete. First indexing/search downloads the small local embedding and reranking weights; the persistent volume caches them for later restarts.

Index the bundled collections once. Through the frontend, open **Bid library → Add a bid → Use an existing backend folder**, use `Bid1` for both identifier and folder, and repeat for `Bid2`. Alternatively, call `POST /index` with `{"folder":"Bid1"}`, then `Bid2`, using `X-API-Key`; wait for each returned job before starting the next. Extract each record through the UI when desired. Additional collections can be uploaded through the same UI.

Your local indexes and run history are not published to Docker Hub. If you need that exact state on Railway, transfer it to the volume separately while the service is stopped; do not bake private state into the image.

## Connect the Vercel frontend

Set its server-side `BACKEND_URL` to the Railway HTTPS URL and `BACKEND_API_KEY` to the same `RFP_API_KEY`. Keep its own `SESSION_SECRET`, demo credentials and public `VITE_SITE_URL` as described in the frontend README. Rebuild/redeploy the frontend after environment changes.

## Local run of the same image

From the backend directory, after filling the minimal `.env`:

```bash
docker volume create bidwell-data
docker run --rm --name bidwell -p 8000:8000 \
  --env-file .env --memory=1g --cpus=1 \
  -v bidwell-data:/var/data \
  [registry-redacted]/rfp-intelligence:2026-10-26-railway
```

For a root-owned fresh bind mount, initialize ownership for UID 10001 or use `--user 0:0` as Railway's UID override does. Do not set local relative `RFP_STORAGE_DIR`, `RFP_MODEL_CACHE_DIR` or `RFP_INPUT_ROOT` values in a container: its existing absolute defaults are correct.

## Published image verification

Both release and latest tags were pushed successfully and resolve to:

```text
sha256:ebfe704bb2923c6377948104e5eb8eb7e5d8a40155f2bf9e9c1e9b54399329d4
```

The release passed 22 container tests under 1 GB RAM / one CPU, plus a fresh-volume index/search/live-answer test and a restart-persistence check. Registry verification and the exact tags are recorded in `artifacts/dockerhub-release.json`. Railway itself has not been deployed by this task.
