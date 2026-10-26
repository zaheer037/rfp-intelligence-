"""Small CPU ONNX embedding/reranking models; no PyTorch or hosted embedding calls."""

import threading

import numpy as np

from rfp.core.config import Settings
from rfp.core.observability import stage


class LocalEmbeddings:
    """Quantized BGE-small embeddings, loaded lazily and shared across requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.fingerprint = f"{settings.embedding_model}:{settings.embedding_max_tokens}:fastembed-v1"
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """Return finite, L2-normalized float32 vectors with bounded CPU batches."""
        if not texts:
            raise ValueError("Embedding input must not be empty")
        with self._lock, stage("embedding"):
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self.settings.embedding_model,
                    cache_dir=str(self.settings.model_cache_dir),
                    threads=self.settings.model_threads,
                    max_length=self.settings.embedding_max_tokens,
                    providers=["CPUExecutionProvider"],
                )
            method = self._model.query_embed if query else self._model.passage_embed
            vectors = np.asarray(
                list(method(texts, batch_size=self.settings.embedding_batch_size)), dtype="float32"
            )
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError("Embedding model returned invalid vectors")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Embedding model returned zero vectors")
            return vectors / norms


class LocalReranker:
    """TinyBERT ONNX cross-encoder (~4MB weights) ranks query/passage pairs locally."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage in its original position."""
        if not passages:
            return []
        with self._lock, stage("reranking"):
            if self._model is None:
                import onnxruntime as ort
                from flashrank import Ranker
                from flashrank.Config import model_file_map

                model = Ranker(
                    model_name=self.settings.reranker_model,
                    cache_dir=str(self.settings.model_cache_dir / "reranker"),
                    max_length=512,
                    log_level="WARNING",
                )
                # FlashRank does not expose session options; replace its default CPU thread pool.
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.settings.model_threads
                options.inter_op_num_threads = 1
                model.session = ort.InferenceSession(
                    str(model.model_dir / model_file_map[self.settings.reranker_model]),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._model = model
            from flashrank import RerankRequest

            scores = [0.0] * len(passages)
            # Small batches bound cross-encoder activation memory on a 1GB host.
            for start in range(0, len(passages), 4):
                ranked = self._model.rerank(
                    RerankRequest(
                        query=query,
                        passages=[
                            {"id": i, "text": text}
                            for i, text in enumerate(passages[start : start + 4], start)
                        ],
                    )
                )
                for hit in ranked:
                    scores[hit["id"]] = float(hit["score"])
            return scores
