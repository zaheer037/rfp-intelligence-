FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RFP_ENVIRONMENT=production \
    HF_HOME=/var/data/models RFP_MODEL_CACHE_DIR=/var/data/models RFP_STORAGE_DIR=/var/data/storage RFP_INPUT_ROOT=/app/bids \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /app/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 120 --retries 5 -r requirements.lock
COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 rfp \
    && mkdir -p /var/data /app/bids && chown -R rfp:rfp /var/data /app
COPY artifacts/retrieval-evaluation.json /app/artifacts/retrieval-evaluation.json
COPY bids/Bid1 /app/bids/Bid1
COPY bids/Bid2 /app/bids/Bid2
USER rfp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health/ready')"
CMD ["rfp", "serve"]
