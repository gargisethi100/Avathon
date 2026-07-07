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


def _chunk_index(chunk_id: str):
    tail = chunk_id.rsplit("::", 1)[-1]
    return int(tail) if tail.isdigit() else None


def search_traced(index, contract_id, query_vec, query_tokens, mode,
                  k: int = 10, pool: int = 20, reranker=None, query_text=None):
    """Like search(), but ALSO returns a full per-chunk retrieval trace.

    Returns (hits, records). `hits` is identical to search(...) for the same mode
    (so answers/metrics are unchanged); `records` is a per-chunk breakdown over the
    fused candidate pool — dense/bm25/rrf/reranker rank+score, final_rank, selected —
    the always-on backend trace the UI diagnostics render.
    """
    dense = index.dense_search(contract_id, query_vec, pool)
    sparse = index.bm25_search(contract_id, query_tokens, pool)
    fused = reciprocal_rank_fusion([dense, sparse])
    reranked = None
    if mode == "hybrid_rerank":
        if reranker is None:
            raise ValueError("hybrid_rerank requires a reranker")
        reranked = reranker.rerank(query_text, fused[:pool])

    final = {"dense": dense, "bm25": sparse, "hybrid": fused,
             "hybrid_rerank": reranked}.get(mode)
    if final is None:
        raise ValueError(f"unknown retrieval mode: {mode!r}")
    final = final[:k]

    def rankmap(lst):
        return {c.chunk_id: (i + 1, float(s)) for i, (c, s) in enumerate(lst)}

    d, b, f = rankmap(dense), rankmap(sparse), rankmap(fused)
    r = rankmap(reranked) if reranked is not None else {}
    final_rank = {c.chunk_id: i + 1 for i, (c, _) in enumerate(final)}
    selected = set(final_rank)

    by_id = {}
    for lst in (fused, dense, sparse):
        for c, _ in lst:
            by_id.setdefault(c.chunk_id, c)

    records = []
    for cid, ch in by_id.items():
        records.append({
            "chunk_id": cid, "chunk_index": _chunk_index(cid),
            "contract_id": getattr(ch, "contract_id", contract_id),
            "text": ch.text, "char_start": ch.char_start, "char_end": ch.char_end,
            "dense_rank": d.get(cid, (None, None))[0], "dense_score": d.get(cid, (None, None))[1],
            "bm25_rank": b.get(cid, (None, None))[0], "bm25_score": b.get(cid, (None, None))[1],
            "rrf_rank": f.get(cid, (None, None))[0], "rrf_score": f.get(cid, (None, None))[1],
            "reranker_rank": r.get(cid, (None, None))[0], "reranker_score": r.get(cid, (None, None))[1],
            "final_rank": final_rank.get(cid), "selected": cid in selected,
        })
    records.sort(key=lambda x: (x["final_rank"] is None, x["final_rank"] or 1e9, x["rrf_rank"] or 1e9))
    return final, records


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
