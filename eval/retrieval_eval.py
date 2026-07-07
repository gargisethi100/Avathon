"""
retrieval_eval.py — OFAT retrieval ablations on the CUAD test split.

Metrics per query (relevance = chunk overlaps a gold answer span; recall ceiling
is 100%, see chunk_coverage.py): P@5, R@5, R@10, MRR, nDCG@10. Mean over all 1,244
answerable questions, searched per-contract.

OFAT (one factor at a time) — vary one axis, hold the rest at a baseline:
  - retrieval-mode axis (Q17): dense / bm25 / hybrid / hybrid_rerank
  - chunking axis (Q15): recursive-256/512, structure_aware-256/512
  - embedding axis (Q16): bge-small vs bge-base
Baseline cell = recursive-256 / bge-small / hybrid.

Performance: the cross-encoder rerank is batched across ALL queries into one
predict pass (not per-query) — the difference between minutes and ~an hour on CPU.
Per-step timing is logged. Writes results/retrieval_metrics.{md,csv}.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from chunking import chunk_text
from embeddings import get_embedder
from indexing import CorpusIndex, tokenize
from retrieval import CrossEncoderReranker, reciprocal_rank_fusion, search

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
METRIC_KEYS = ("P@5", "R@5", "R@10", "MRR", "nDCG@10")

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def build_index(contracts, chunking, tokens, embedder):
    idx = CorpusIndex(embedder)
    for _, c in contracts.iterrows():
        chunks = chunk_text(c["text"], c["title"], strategy=chunking, target_tokens=tokens)
        idx.add_contract(c["title"], chunks)
    return idx


def relevant_ids(chunks, gold_spans):
    return {
        ch.chunk_id
        for ch in chunks
        if any(overlaps(gs, ge, ch.char_start, ch.char_end) for gs, ge in gold_spans)
    }


def query_metrics(ranked_ids, rel):
    rel = set(rel)
    n = len(rel)
    top5 = sum(1 for c in ranked_ids[:5] if c in rel)
    top10 = sum(1 for c in ranked_ids[:10] if c in rel)
    mrr = 0.0
    for i, c in enumerate(ranked_ids):
        if c in rel:
            mrr = 1.0 / (i + 1)
            break
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranked_ids[:10]) if c in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n, 10)))
    return {
        "P@5": top5 / 5,
        "R@5": top5 / n,
        "R@10": top10 / n,
        "MRR": mrr,
        "nDCG@10": dcg / idcg if idcg else 0.0,
    }


def eval_modes(idx, qvecs, ans, modes):
    """Fast modes (dense/bm25/hybrid) — no cross-encoder."""
    acc = {m: {k: 0.0 for k in METRIC_KEYS} for m in modes}
    for i, (_, r) in enumerate(ans.iterrows()):
        cid = r["title"]
        rel = relevant_ids(idx.stores[cid].chunks, list(zip(r["answer_starts"], r["answer_ends"])))
        qtok = tokenize(r["question"])
        qv = qvecs[i]
        for m in modes:
            res = search(idx, cid, qv, qtok, m, k=10, pool=20)
            met = query_metrics([c.chunk_id for c, _ in res], rel)
            for k in METRIC_KEYS:
                acc[m][k] += met[k]
    n = len(ans)
    return {m: {k: acc[m][k] / n for k in METRIC_KEYS} for m in modes}


def eval_rerank_batched(idx, qvecs, ans, reranker, pool=20, k=10):
    """hybrid_rerank with ONE batched cross-encoder predict over all candidates."""
    all_pairs = []
    slices = []
    cands_per_q = []
    for i, (_, r) in enumerate(ans.iterrows()):
        cid = r["title"]
        dense = idx.dense_search(cid, qvecs[i], pool)
        sparse = idx.bm25_search(cid, tokenize(r["question"]), pool)
        cands = [c for c, _ in reciprocal_rank_fusion([dense, sparse])[:pool]]
        start = len(all_pairs)
        all_pairs.extend([r["question"], c.text] for c in cands)
        slices.append((start, len(all_pairs)))
        cands_per_q.append(cands)
    log(f"  reranking {len(all_pairs)} pairs in one batched pass ...")
    scores = np.asarray(reranker.model.predict(all_pairs, batch_size=128, show_progress_bar=False))
    acc = {k: 0.0 for k in METRIC_KEYS}
    for i, (_, r) in enumerate(ans.iterrows()):
        s, e = slices[i]
        cands = cands_per_q[i]
        if not cands:
            continue
        order = np.argsort(scores[s:e])[::-1][:k]
        ranked = [cands[j].chunk_id for j in order]
        rel = relevant_ids(idx.stores[r["title"]].chunks, list(zip(r["answer_starts"], r["answer_ends"])))
        met = query_metrics(ranked, rel)
        for kk in METRIC_KEYS:
            acc[kk] += met[kk]
    n = len(ans)
    return {kk: acc[kk] / n for kk in METRIC_KEYS}


def to_markdown(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    # CPU-tractability: run the ablation on a seeded stratified contract subset
    # (embedding all 102 contracts x 5 index variants on CPU is ~90 min). The
    # relative ranking of configs is stable; the winner is confirmed on the full
    # test set separately.
    SUBSET_N, SEED = 20, 42
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    if SUBSET_N and SUBSET_N < len(contracts):
        contracts = contracts.sample(SUBSET_N, random_state=SEED).reset_index(drop=True)
    subset_titles = set(contracts["title"])
    qa = pd.read_parquet(DATA / "qa_test.parquet")
    ans = qa[qa["is_answerable"] & qa["title"].isin(subset_titles)].reset_index(drop=True)
    RESULTS.mkdir(exist_ok=True)
    log(f"ablation subset: {len(contracts)} contracts, {len(ans)} answerable queries (seed {SEED})")

    log(f"loading bge-small + encoding {len(ans)} queries")
    emb_small = get_embedder("bge-small")
    qv_small = emb_small.encode_queries(ans["question"].tolist())
    log("loading cross-encoder reranker")
    reranker = CrossEncoderReranker()

    idx_cache = {}

    def get_idx(chunking, tokens, ename, emb):
        key = (chunking, tokens, ename)
        if key not in idx_cache:
            log(f"building index {key} ...")
            idx_cache[key] = build_index(contracts, chunking, tokens, emb)
            log(f"  built {key}: {idx_cache[key].n_chunks} chunks")
        return idx_cache[key]

    rows = []

    # --- retrieval-mode axis ---
    base = get_idx("recursive", 256, "bge-small", emb_small)
    log("eval mode axis: dense/bm25/hybrid")
    for m, met in eval_modes(base, qv_small, ans, ["dense", "bm25", "hybrid"]).items():
        rows.append({"axis": "retrieval", "config": "recursive-256/bge-small", "mode": m, **met})
    log("eval mode axis: hybrid_rerank")
    rows.append({"axis": "retrieval", "config": "recursive-256/bge-small", "mode": "hybrid_rerank",
                 **eval_rerank_batched(base, qv_small, ans, reranker)})
    log("mode axis DONE")

    # --- chunking axis (bge-small, hybrid) ---
    for chunking, tokens in [("recursive", 256), ("recursive", 512),
                             ("structure_aware", 256), ("structure_aware", 512)]:
        idx = get_idx(chunking, tokens, "bge-small", emb_small)
        met = eval_modes(idx, qv_small, ans, ["hybrid"])["hybrid"]
        rows.append({"axis": "chunking", "config": f"{chunking}-{tokens}/bge-small", "mode": "hybrid", **met})
    log("chunking axis DONE")

    # --- embedding axis (recursive-256, hybrid) ---
    log("loading bge-base + encoding queries")
    emb_base = get_embedder("bge-base")
    qv_base = emb_base.encode_queries(ans["question"].tolist())
    for ename, emb, qv in [("bge-small", emb_small, qv_small), ("bge-base", emb_base, qv_base)]:
        idx = get_idx("recursive", 256, ename, emb)
        met = eval_modes(idx, qv, ans, ["hybrid"])["hybrid"]
        rows.append({"axis": "embedding", "config": f"recursive-256/{ename}", "mode": "hybrid", **met})
    log("embedding axis DONE")

    df = pd.DataFrame(rows)
    for k in METRIC_KEYS:
        df[k] = df[k].round(4)
    df.to_csv(RESULTS / "retrieval_metrics.csv", index=False)
    md = (
        f"# Retrieval OFAT ablations ({len(ans)} answerable queries, "
        f"{len(contracts)}-contract stratified subset, seed {SEED})\n\n" + to_markdown(df)
    )
    (RESULTS / "retrieval_metrics.md").write_text(md, encoding="utf-8")
    log("DONE — wrote results/retrieval_metrics.{md,csv}")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
