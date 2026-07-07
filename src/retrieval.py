"""
retrieval.py — retrieval modes: dense / BM25 / hybrid-RRF / +cross-encoder rerank.

These are the retrieval-mode axis of the Phase-3 ablation (Q17). All operate on a
single contract's index (per-contract scoping).

Reciprocal Rank Fusion (RRF): fuse two ranked lists by summing 1/(k+rank). It needs
no score normalization across the incomparable cosine (dense) and BM25 (sparse)
scales — that scale-invariance is exactly why RRF beats naive score addition for
hybrid legal retrieval.

Cross-encoder rerank: a bi-encoder scores query and passage independently (fast,
approximate); a cross-encoder attends over the (query, passage) pair jointly (slow,
precise). We rerank only the fused top-`pool` candidates → precision at the top for
a bounded compute cost.
"""
from __future__ import annotations

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass


def reciprocal_rank_fusion(ranked_lists, k: int = 60):
    """Fuse ranked lists of (chunk, score). Returns [(chunk, rrf_score)] desc."""
    scores: dict[str, float] = {}
    by_id: dict[str, object] = {}
    for lst in ranked_lists:
        for rank, (chunk, _score) in enumerate(lst):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            by_id[cid] = chunk
    order = sorted(scores.items(), key=lambda x: -x[1])
    return [(by_id[cid], s) for cid, s in order]


class CrossEncoderReranker:
    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        # Cap sequence length: query + a ~256-token legal chunk fits well under 384,
        # and capping bounds the per-pair CPU cost on the long tail.
        self.model = CrossEncoder(model_id, max_length=384)
        self.name = model_id

    def rerank(self, query: str, candidates):
        """candidates: [(chunk, score)] -> reordered by cross-encoder relevance."""
        if not candidates:
            return []
        pairs = [[query, c.text] for c, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1]
        return [(candidates[i][0], float(scores[i])) for i in order]


def search(
    index,
    contract_id: str,
    query_vec,
    query_tokens,
    mode: str,
    k: int = 10,
    pool: int = 20,
    reranker: "CrossEncoderReranker | None" = None,
    query_text: str | None = None,
):
    """Return ranked [(chunk, score)] for one query in one contract."""
    if mode == "dense":
        return index.dense_search(contract_id, query_vec, k)
    if mode == "bm25":
        return index.bm25_search(contract_id, query_tokens, k)
    if mode in ("hybrid", "hybrid_rerank"):
        dense = index.dense_search(contract_id, query_vec, pool)
        sparse = index.bm25_search(contract_id, query_tokens, pool)
        fused = reciprocal_rank_fusion([dense, sparse])
        if mode == "hybrid":
            return fused[:k]
        if reranker is None:
            raise ValueError("hybrid_rerank requires a reranker")
        return reranker.rerank(query_text, fused[:pool])[:k]
    raise ValueError(f"unknown retrieval mode: {mode!r}")


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
