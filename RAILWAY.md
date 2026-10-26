# Deploy Zaheer's RFP Intelligence backend on Railway

Source: [zaheer037/rfp-intelligence-](https://github.com/zaheer037/rfp-intelligence-). This guide builds the Docker image directly from that repository, using its root Dockerfile.

## Create the service

1. Connect Railway to GitHub and grant the Railway GitHub app access to **`zaheer037/rfp-intelligence-`**, including the final hyphen.
2. Select that repository and the `main` branch. Keep the root directory at the repository root; Railway detects `Dockerfile` there.
3. Review the **Apply changes → Details** banner, then click its global **Deploy** button to apply staged service creation. The service-level deploy button does not apply an uncommitted changeset.

See Railway's [service sources](https://docs.railway.com/services) and [staged changes](https://docs.railway.com/deployments/staged-changes). Docker Hub is not needed for this source-based deployment.

## Storage, credentials and port

Attach one persistent volume at **`/var/data`**. The image stores models under `/var/data/models` and indexes/jobs/uploads/traces under `/var/data/storage`.

Set the two application credentials:

```dotenv
RFP_API_KEY=<your strong backend key, at least 24 characters>
GROQ_API_KEY=<your Groq key>
```

Set **`RAILWAY_RUN_UID=0`** for Railway's root-owned volume; the image otherwise defaults to UID 10001. This platform setting avoids write-permission errors. See [Railway volume permissions](https://docs.railway.com/volumes/reference).

The server listens on `0.0.0.0:$PORT`, defaulting to 8000. Railway supplies `PORT`; its public domain target must use the same value. Set the readiness healthcheck to **`/health/ready`** and keep the image's default start command. Use one replica and the tested 1 GB RAM / one CPU budget. Generate a public HTTPS domain; its platform-provided hostname automatically configures A2A discovery.

## First indexing

A fresh volume contains no indexes or model cache. First inference downloads the small ONNX embedding/reranker models. Readiness means the API is running, not that indexing or Groq requests are complete.

Index Bid1, then Bid2 sequentially. In the frontend, open **Bid library → Add a bid → Use an existing backend folder**, using `Bid1` for identifier/folder and then `Bid2`. Alternatively, submit `POST /index` with `{"folder":"Bid1"}`, then `Bid2`, and the `X-API-Key` header; wait for each job. Extract each record when needed for structured fields/comparison.

## Frontend connection

Deploy [zaheer037/rfp-frontend](https://github.com/zaheer037/rfp-frontend) to Vercel. Its server-side `BACKEND_URL` is the Railway HTTPS origin, and `BACKEND_API_KEY` matches `RFP_API_KEY`. Set its own `SESSION_SECRET` and public `VITE_SITE_URL` as described in its deployment guide. Never put backend credentials in browser-prefixed variables.

## Local Docker verification

```bash
docker build -t rfp-intelligence .
docker volume create rfp-data
docker run --rm -p 8000:8000 --env-file .env \
  -v rfp-data:/var/data rfp-intelligence
```

The local release passed 22 container tests including OCR under 1 GB / one CPU, plus fresh-volume indexing/search/live-answer and restart-persistence checks. These are local measurements, not proof of successful hosted deployment. For a crash, open the deployment's logs and inspect its startup traceback before changing the image.
