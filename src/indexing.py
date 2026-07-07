"""
indexing.py — per-contract FAISS flat + BM25 indexes.

Retrieval is per-contract scoped (the enterprise flow: pick a document, ask
questions), so each contract gets its own tiny exact index. Consequences:
  - FAISS IndexFlatIP over L2-normalized vectors == exact cosine search; at
    ~60-200 chunks/contract, ANN would only add error for no speed win (D-06).
  - Adding a new contract touches only that contract's index — the "keep the
    index fresh without re-embedding the corpus" story (freshness).

`dense_search` takes a pre-encoded query vector; `bm25_search` takes query tokens.
Fusion / re-ranking live in retrieval.py (Phase 3); this module is the primitives.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from chunking import Chunk

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class ContractStore:
    contract_id: str
    chunks: list[Chunk]
    vecs: np.ndarray  # (n, dim) L2-normalized float32
    faiss_index: object = None
    bm25: object = None

    def build(self) -> "ContractStore":
        self.faiss_index = faiss.IndexFlatIP(self.vecs.shape[1])
        self.faiss_index.add(self.vecs)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        return self


class CorpusIndex:
    """Collection of per-contract stores sharing one embedder."""

    def __init__(self, embedder=None, embedder_name: str | None = None):
        self.embedder = embedder
        self.embedder_name = embedder_name or (embedder.name if embedder else None)
        self.stores: dict[str, ContractStore] = {}

    # ---- build ----
    def add_contract(self, contract_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode_passages([c.text for c in chunks])
        self.stores[contract_id] = ContractStore(contract_id, list(chunks), vecs).build()

    # ---- search primitives (per contract) ----
    def dense_search(self, contract_id: str, query_vec: np.ndarray, k: int = 10):
        s = self.stores[contract_id]
        qv = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        d, idx = s.faiss_index.search(qv, min(k, len(s.chunks)))
        return [(s.chunks[i], float(d[0][r])) for r, i in enumerate(idx[0]) if i != -1]

    def bm25_search(self, contract_id: str, query_tokens: list[str], k: int = 10):
        s = self.stores[contract_id]
        scores = s.bm25.get_scores(query_tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(s.chunks[i], float(scores[i])) for i in order]

    # ---- persistence (indexes/, gitignored — rebuilt from vecs on load) ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "embedder_name": self.embedder_name,
            "stores": {
                cid: {"chunks": [c.to_dict() for c in s.chunks], "vecs": s.vecs}
                for cid, s in self.stores.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        idx = cls(embedder=None, embedder_name=blob["embedder_name"])
        for cid, s in blob["stores"].items():
            chunks = [Chunk(**c) for c in s["chunks"]]
            idx.stores[cid] = ContractStore(cid, chunks, s["vecs"]).build()
        return idx

    @property
    def n_chunks(self) -> int:
        return sum(len(s.chunks) for s in self.stores.values())

    @property
    def contract_ids(self) -> list[str]:
        return list(self.stores)
