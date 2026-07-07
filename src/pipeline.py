"""
pipeline.py — end-to-end ClauseLens orchestration.

    query -> query gate -> (retrieve -> generate -> verify) -> answer + trace

Config-driven so the Phase-3 winning retrieval config plugs straight in. Every
answer carries a trace (gate label, retrieved chunk ids + scores, quotes,
verifier verdict) — the observability the demo and error analysis rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import chunk_text
from embeddings import get_embedder
from generation import BedrockGenerator
from indexing import CorpusIndex, tokenize
from query_gate import QueryGate
from retrieval import CrossEncoderReranker, search
from verifier import should_abstain, verify_answer

# Default config; updated to the Phase-3 winner once measured.
DEFAULTS = dict(chunking="recursive", tokens=256, embedder="bge-small",
                mode="hybrid_rerank", abstain_threshold=None)


class ClauseLens:
    def __init__(self, use_gate: bool = True, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self.embedder = get_embedder(self.cfg["embedder"])
        self.index = CorpusIndex(self.embedder)
        self.reranker = CrossEncoderReranker() if "rerank" in self.cfg["mode"] else None
        self.generator = BedrockGenerator()
        self.gate = QueryGate() if use_gate else None

    def add_contract(self, contract_id: str, text: str) -> None:
        chunks = chunk_text(text, contract_id, self.cfg["chunking"], self.cfg["tokens"])
        self.index.add_contract(contract_id, chunks)

    def answer(self, question: str, contract_id: str, k: int = 5) -> dict:
        trace = {"question": question, "contract_id": contract_id}

        if self.gate is not None:
            label = self.gate.classify(question)
            trace["gate"] = label
            if label == "OUT_OF_SCOPE":
                return {"status": "declined_out_of_scope",
                        "answer": "That question isn't about this contract, so I can't answer it from the document.",
                        "trace": trace}
            if label == "AMBIGUOUS":
                return {"status": "needs_clarification",
                        "answer": "That's ambiguous for this contract — which clause or detail do you mean (e.g., which date, which party, which type of termination)?",
                        "trace": trace}

        qv = self.embedder.encode_queries([question])[0]
        hits = search(self.index, contract_id, qv, tokenize(question), self.cfg["mode"],
                      k=k, pool=20, reranker=self.reranker, query_text=question)
        trace["retrieved"] = [(c.chunk_id, round(s, 3)) for c, s in hits]

        thr = self.cfg["abstain_threshold"]
        if thr is not None and should_abstain(hits, thr):
            trace["verdict"] = "PASS"
            return {"status": "abstained_low_score", "answer": "NOT_FOUND",
                    "abstained": True, "trace": trace}

        ans = self.generator.answer(question, hits)
        ver = verify_answer(ans, hits)
        trace["verdict"] = ver.verdict
        return {
            "status": "answered",
            "answer": ans.text,
            "abstained": ans.abstained,
            "cited_chunk_ids": ans.cited_chunk_ids,
            "quotes": ans.quotes,
            "verdict": ver.verdict,
            "passed": ver.passed,
            "usage": ans.usage,
            "trace": trace,
        }
