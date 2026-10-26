"""Hybrid, filter-aware passage search usable directly and as an ADK tool."""

import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rfp.core.observability import INDEX_CHUNKS, SEARCH_HITS, SEARCHES, event, stage
from rfp.schemas import Chunk, SearchHit
from rfp.utils.identifiers import validate_bid_id

SYNONYMS = {
    "deadline": "submission due date closing date",
    "warranty": "OEM extended warranty support years",
    "bond": "bid security surety bond percentage",
    "affidavit": "required documentation certification forms",
    "payment": "invoice invoicing net terms",
    "installation": "deployment imaging white glove setup",
    "phone": "agency POC procurement contact telephone",
    "term": "initial contract duration years renewal extension",
    "delivery": "shipping award days delivery",
    "specifications": "CPU processor RAM memory storage display",
}


def tokenize(text: str) -> list[str]:
    """Preserve exact hyphenated model/SKU tokens and also index their component words."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    stopwords = {
        "a",
        "an",
        "the",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "must",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "as",
        "and",
        "or",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "you",
        "your",
        "our",
    }
    return [t for t in tokens if t not in stopwords] + [
        part for token in tokens if "-" in token for part in token.split("-")
    ]


@dataclass(frozen=True)
class Corpus:
    chunks: list[Chunk]
    vectors: np.ndarray
    index: object
    bm25: BM25Okapi


class Retrieval:
    """Combine dense vectors and BM25 with RRF, then cross-encoder rerank candidates."""

    def __init__(self, indexing, embeddings, reranker):
        self.indexing = indexing
        self.embeddings = embeddings
        self.reranker = reranker
        self._corpora: dict[str, Corpus] = {}
        self.unavailable_bids: dict[str, str] = {}
        self._lock = threading.RLock()

    def reload(self, bid_id: str | None = None) -> None:
        """Load committed snapshots; replace complete in-memory corpora under one lock."""
        ids = (
            [validate_bid_id(bid_id)]
            if bid_id
            else [p.name for p in self.indexing.root.iterdir() if p.is_dir()]
        )
        for identifier in ids:
            try:
                loaded = self.indexing.load(identifier)
            except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError) as exc:
                with self._lock:
                    self._corpora.pop(identifier, None)
                    self.unavailable_bids[identifier] = (
                        "Reindex required: incompatible or unreadable snapshot"
                    )
                event("snapshot_unavailable", bid_id=identifier, error_type=type(exc).__name__)
                continue
            if loaded:
                chunks, index, _ = loaded
                vectors = index.reconstruct_n(0, index.ntotal)
                corpus = Corpus(chunks, vectors, index, BM25Okapi([tokenize(c.text) for c in chunks]))
                with self._lock:
                    self._corpora[identifier] = corpus
                    self.unavailable_bids.pop(identifier, None)
        INDEX_CHUNKS.set(sum(len(c.chunks) for c in self._corpora.values()))

    def bid_ids(self) -> list[str]:
        """Return currently available bids without exposing local paths."""
        with self._lock:
            return sorted(self._corpora)

    def chunks(self, bid_ids: list[str]) -> dict[str, Chunk]:
        """Expose an evidence registry for deterministic citation validation."""
        with self._lock:
            return {
                c.chunk_id: c for bid in bid_ids if bid in self._corpora for c in self._corpora[bid].chunks
            }

    def search(
        self,
        query: str,
        *,
        bid_id: str | None = None,
        doc_type: str | None = None,
        addendum_number: int | None = None,
        top_k: int = 5,
        mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    ) -> list[SearchHit]:
        """Restrict the corpus first, fuse candidates and return cited ranked passages."""
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 50:
            raise ValueError("Search requires a nonempty query <=2000 characters and top_k 1–50")
        if mode not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("Invalid retrieval mode")
        if bid_id:
            validate_bid_id(bid_id)
        SEARCHES.labels(mode).inc()
        with stage("retrieval"):
            with self._lock:
                corpora = [(k, v) for k, v in self._corpora.items() if not bid_id or k == bid_id]
            if bid_id and not corpora:
                raise ValueError(f"Bid is not indexed: {bid_id}")
            expanded = (
                query
                + " "
                + " ".join(
                    v
                    for k, v in SYNONYMS.items()
                    if k in query.lower() or (k == "warranty" and "warrant" in query.lower())
                )
            )
            # Query references can narrow an otherwise unscoped search, including multi-bid comparisons.
            mentioned = [k for k, _ in corpora if re.search(rf"\b{re.escape(k)}\b", query, re.I)]
            if not bid_id and mentioned:
                corpora = [(k, v) for k, v in corpora if k in mentioned]
            candidates = []
            for _, corpus in corpora:
                positions = [
                    i
                    for i, c in enumerate(corpus.chunks)
                    if (doc_type is None or c.doc_type == doc_type)
                    and (addendum_number is None or c.addendum_number == addendum_number)
                ]
                if positions:
                    candidates.append((corpus, positions))
            if not candidates:
                SEARCH_HITS.observe(0)
                return []
            vector = self.embeddings.encode([expanded], query=True)
            faiss.normalize_L2(vector)
            hits = []
            for corpus, positions in candidates:
                # ponytail: exact filtered subset is O(n); use FAISS selectors/partitioning at large scale.
                subset = np.ascontiguousarray(corpus.vectors[positions], dtype="float32")
                index = faiss.IndexFlatIP(subset.shape[1])
                index.add(subset)
                limit = min(len(positions), max(40, top_k * 4))
                dense_scores, dense_ids = index.search(vector, limit)
                dense = {
                    positions[int(i)]: (rank, float(score))
                    for rank, (i, score) in enumerate(zip(dense_ids[0], dense_scores[0], strict=True), 1)
                }
                keyword_scores = corpus.bm25.get_scores(tokenize(expanded))
                sparse_order = sorted(positions, key=lambda i: float(keyword_scores[i]), reverse=True)[:limit]
                sparse = {
                    i: (rank, float(keyword_scores[i]))
                    for rank, i in enumerate(sparse_order, 1)
                    if keyword_scores[i] > 0
                }
                ids = set(dense) if mode == "dense" else set(dense) | set(sparse)
                for i in ids:
                    score = (
                        dense[i][1]
                        if mode == "dense"
                        else sum(
                            weight / (60 + ranks[i][0])
                            for ranks, weight in ((dense, self.indexing.settings.dense_weight), (sparse, 1))
                            if i in ranks
                        )
                    )
                    hits.append(
                        SearchHit(
                            chunk=corpus.chunks[i],
                            score=score,
                            dense_score=dense.get(i, (0, None))[1],
                            keyword_score=sparse.get(i, (0, None))[1],
                        )
                    )
            hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
            # ponytail: near-duplicate removal is O(candidates²), bounded to the small candidate pool.
            distinct = []
            for hit in hits:
                terms = set(tokenize(hit.chunk.text))
                duplicate = False
                for prior, prior_terms in distinct:
                    if (hit.chunk.bid_id, hit.chunk.file_name, hit.chunk.page_number) == (
                        prior.chunk.bid_id,
                        prior.chunk.file_name,
                        prior.chunk.page_number,
                    ) and len(terms & prior_terms) / max(1, len(terms | prior_terms)) > 0.8:
                        duplicate = True
                        break
                if not duplicate:
                    distinct.append((hit, terms))
                if len(distinct) >= max(40, top_k * 4):
                    break
            hits = [hit for hit, _ in distinct]
            if mode == "hybrid_rerank" and hits:
                scores = self.reranker.score(query, [h.chunk.text for h in hits])
                for hit, score in zip(hits, scores, strict=True):
                    hit.rerank_score = score
                prior = {h.chunk.chunk_id: rank for rank, h in enumerate(hits, 1)}
                cross = {
                    h.chunk.chunk_id: rank
                    for rank, h in enumerate(sorted(hits, key=lambda h: (-h.rerank_score, -h.score)), 1)
                }
                weight = self.indexing.settings.reranker_weight
                for hit in hits:
                    # A tiny cross-encoder is a weak signal; preserve the strong retrieval prior.
                    hit.score = (1 - weight) / (60 + prior[hit.chunk.chunk_id]) + weight / (
                        60 + cross[hit.chunk.chunk_id]
                    )
                hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
            unique, seen = [], set()
            for hit in hits:
                key = (hit.chunk.bid_id, hit.chunk.content_hash)
                if key not in seen:
                    seen.add(key)
                    unique.append(hit)
                if len(unique) == top_k:
                    break
            SEARCH_HITS.observe(len(unique))
            return unique
