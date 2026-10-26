"""Reproducible passage-level retrieval evaluation; never fabricate missing model results."""

import argparse
import json
import time
from pathlib import Path

from rfp.agents.validation import normalized
from rfp.core.config import Settings
from rfp.ingestion.indexing import Indexing
from rfp.search.models import LocalEmbeddings, LocalReranker
from rfp.search.retrieval import Retrieval


def evaluate(retrieval, examples, k=5):
    """Score Recall@k and MRR@k against all labeled relevant passage locations."""
    output = []
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        recalls, reciprocal, details = [], [], []
        start = time.perf_counter()
        for row in examples:
            hits = retrieval.search(row["question"], bid_id=row["bid_id"], top_k=k, mode=mode)
            # Labels use filename/page plus a passage substring, avoiding incidental chunk IDs.
            matches = []
            for rank, hit in enumerate(hits, 1):
                for i, expected in enumerate(row["expected"]):
                    if (
                        hit.chunk.file_name == expected["file"]
                        and hit.chunk.page_number == expected["page"]
                        and normalized(expected["passage"]) in normalized(hit.chunk.text)
                    ):
                        matches.append((i, rank))
            recall = len({i for i, _ in matches}) / len(row["expected"])
            rr = 1 / min(rank for _, rank in matches) if matches else 0
            recalls.append(recall)
            reciprocal.append(rr)
            details.append(
                {
                    "question": row["question"],
                    "recall": recall,
                    "reciprocal_rank": rr,
                    "retrieved": [h.chunk.chunk_id for h in hits],
                }
            )
        output.append(
            {
                "configuration": mode,
                f"recall@{k}": sum(recalls) / len(examples),
                f"mrr@{k}": sum(reciprocal) / len(examples),
                "questions": len(examples),
                "duration_seconds": time.perf_counter() - start,
                "details": details,
            }
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/questions.json")
    parser.add_argument("--output", default="artifacts/retrieval-evaluation.json")
    args = parser.parse_args()
    settings = Settings()
    embedding = LocalEmbeddings(settings)
    retrieval = Retrieval(Indexing(settings, embedding), embedding, LocalReranker(settings))
    retrieval.reload()
    result = {
        "embedding": settings.embedding_model,
        "reranker": settings.reranker_model,
        "dense_weight": settings.dense_weight,
        "reranker_weight": settings.reranker_weight,
        "results": evaluate(retrieval, json.loads(Path(args.dataset).read_text())),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
