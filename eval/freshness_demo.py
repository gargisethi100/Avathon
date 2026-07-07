"""
freshness_demo.py — knowledge freshness: grow the corpus WITHOUT re-embedding it.

The corpus grows over time (new contracts arrive). Because ClauseLens keeps a
per-contract store (indexing.CorpusIndex), adding a contract only chunks + embeds
THAT contract — every existing contract's vectors are untouched (verified by
object identity below). The real re-embed trigger is an embedding-model version
change, not corpus growth (write-up §4).

Run:  python eval/freshness_demo.py     -> results/freshness_demo.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from chunking import chunk_text
from embeddings import get_embedder
from indexing import CorpusIndex, tokenize

RESULTS = ROOT / "results"


def main() -> int:
    contracts = pd.read_parquet(ROOT / "data/processed/contracts_test.parquet")
    base, new = contracts.iloc[:3], contracts.iloc[3]

    emb = get_embedder("bge-small")
    idx = CorpusIndex(emb)

    t0 = time.perf_counter()
    for _, c in base.iterrows():
        idx.add_contract(c["title"], chunk_text(c["text"], c["title"], "recursive", 512))
    t_base = time.perf_counter() - t0
    n_before = idx.n_chunks
    # snapshot identity of existing vector arrays — must be untouched by the add
    vec_ids = {cid: id(s.vecs) for cid, s in idx.stores.items()}

    t0 = time.perf_counter()
    idx.add_contract(new["title"], chunk_text(new["text"], new["title"], "recursive", 512))
    t_add = time.perf_counter() - t0

    untouched = all(id(idx.stores[cid].vecs) == vid for cid, vid in vec_ids.items())
    q = "What is the governing law?"
    hits = idx.bm25_search(new["title"], tokenize(q), k=3)

    lines = [
        "# Knowledge-freshness demo — incremental add without re-embedding\n",
        f"- Base index: {len(base)} contracts, {n_before} chunks, built in {t_base:.1f}s",
        f"- **Added 1 new contract in {t_add:.1f}s** ({idx.n_chunks - n_before} new chunks embedded)",
        f"- Existing contracts' vectors untouched (object-identity check): **{untouched}**",
        f"- New contract immediately queryable — top BM25 hit for \"{q}\":",
        f"  > {hits[0][0].text[:180]}…" if hits else "  (no hit)",
        "\nRe-embedding is only required when the embedding model version changes — not when the corpus grows.",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "freshness_demo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
