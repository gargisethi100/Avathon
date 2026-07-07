"""
pipeline.py — end-to-end ClauseLens orchestration.

    query -> query gate -> (retrieve -> generate -> verify) -> answer + full trace

Behavior (retrieval/generation/gating/verification) is unchanged. What's new is
observability: every query ALWAYS produces the complete backend trace —

  trace = {query, contract_id, gate, retrieval_detail[per-chunk dense/bm25/rrf/
           reranker rank+score, final_rank, selected], retrieved (compat), latency,
           pipeline, verdict}

plus a UI-friendly `evidence` list (only the cited/supporting chunks, mapped
deterministically to the answer's [n] citations). The full trace is also appended
to logs/query_traces.jsonl. The UI shows `evidence` by default and the full
`retrieval_detail` only in an optional diagnostics panel.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import chunk_text
from embeddings import get_embedder
from generation import BedrockGenerator
from indexing import CorpusIndex, tokenize
from query_gate import QueryGate
from retrieval import CrossEncoderReranker, search_traced
from verifier import _norm, should_abstain, verify_answer

# Phase-3 winner (measured): recursive-512 + bge-small + hybrid-RRF.
# The MS-MARCO cross-encoder rerank HURT retrieval on legal text, so it's off.
DEFAULTS = dict(chunking="recursive", tokens=512, embedder="bge-small",
                mode="hybrid", abstain_threshold=None)

_LOG_PATH = ROOT / "logs" / "query_traces.jsonl"


class ClauseLens:
    def __init__(self, use_gate: bool = True, log_traces: bool = True, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self.embedder = get_embedder(self.cfg["embedder"])
        self.index = CorpusIndex(self.embedder)
        self.reranker = CrossEncoderReranker() if "rerank" in self.cfg["mode"] else None
        self.generator = BedrockGenerator()
        self.gate = QueryGate() if use_gate else None
        self.log_traces = log_traces

    def add_contract(self, contract_id: str, text: str) -> None:
        chunks = chunk_text(text, contract_id, self.cfg["chunking"], self.cfg["tokens"])
        self.index.add_contract(contract_id, chunks)

    # ---- helpers ----
    def _pipeline_summary(self) -> str:
        steps = ["Dense + BM25", "RRF fusion"]
        if "rerank" in self.cfg["mode"]:
            steps.append("cross-encoder rerank")
        steps += ["grounded generation", "quote verification"]
        return "  →  ".join(steps)

    def _build_evidence(self, ans, hits, records) -> list[dict]:
        """Only the cited/supporting chunks, mapped to answer [n] citations."""
        rec_by_id = {r["chunk_id"]: r for r in records}
        ordered = [c for c, _ in hits]  # display index n -> ordered[n-1]

        # citation order (unique, preserved); fall back to selected chunks that
        # contain a quote if the model quoted without an explicit [n].
        idxs = list(dict.fromkeys(ans.cited_indices))
        sources = [(n, ordered[n - 1]) for n in idxs if 1 <= n <= len(ordered)]
        if not sources and ans.quotes:
            for n, ch in enumerate(ordered, 1):
                if any(_norm(q) in _norm(ch.text) for q in ans.quotes):
                    sources.append((n, ch))

        evidence = []
        for n, ch in sources:
            rec = rec_by_id.get(ch.chunk_id, {})
            evidence.append({
                "source": n,
                "chunk_id": ch.chunk_id,
                "chunk_index": rec.get("chunk_index"),
                "contract_id": getattr(ch, "contract_id", None),
                "char_start": ch.char_start, "char_end": ch.char_end,
                "excerpt": ch.text,
                "quotes": [q for q in ans.quotes if _norm(q) in _norm(ch.text)],
                "rrf_rank": rec.get("rrf_rank"), "rrf_score": rec.get("rrf_score"),
                "reranker_score": rec.get("reranker_score"),
                "final_rank": rec.get("final_rank"),
            })
        return evidence

    def _finalize(self, result, timings, t0, question, contract_id) -> dict:
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        tr = result.setdefault("trace", {})
        tr.setdefault("query", question)
        tr.setdefault("contract_id", contract_id)
        tr["latency"] = timings
        if self.log_traces:
            self._log_trace(result)
        return result

    def _log_trace(self, result) -> None:
        try:
            tr = result.get("trace", {})
            record = {
                "query": tr.get("query"),
                "contract_id": tr.get("contract_id"),
                "gate": tr.get("gate"),
                "status": result.get("status"),
                "retrieval_detail": tr.get("retrieval_detail"),
                "answer": result.get("answer"),
                "citations": result.get("cited_chunk_ids"),
                "quotes": result.get("quotes"),
                "verdict": result.get("verdict"),
                "latency": tr.get("latency"),
            }
            _LOG_PATH.parent.mkdir(exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # logging must never break a query

    # ---- main ----
    def answer(self, question: str, contract_id: str, k: int = 5) -> dict:
        clock = time.perf_counter
        t0 = clock()
        timings: dict = {}
        trace = {"query": question, "contract_id": contract_id}

        if self.gate is not None:
            g0 = clock()
            label = self.gate.classify(question)
            timings["gate_ms"] = round((clock() - g0) * 1000, 1)
            trace["gate"] = label
            if label == "OUT_OF_SCOPE":
                return self._finalize({
                    "status": "declined_out_of_scope",
                    "answer": "That question isn't about this contract, so I can't answer it from the document.",
                    "trace": trace}, timings, t0, question, contract_id)
            if label == "AMBIGUOUS":
                return self._finalize({
                    "status": "needs_clarification",
                    "answer": "That's ambiguous for this contract — which clause or detail do you mean (e.g., which date, which party, which type of termination)?",
                    "trace": trace}, timings, t0, question, contract_id)

        r0 = clock()
        qv = self.embedder.encode_queries([question])[0]
        hits, records = search_traced(
            self.index, contract_id, qv, tokenize(question), self.cfg["mode"],
            k=k, pool=20, reranker=self.reranker, query_text=question)
        timings["retrieval_ms"] = round((clock() - r0) * 1000, 1)
        trace["retrieval_detail"] = records
        trace["retrieved"] = [(r["chunk_id"], round(r["rrf_score"] or r["dense_score"] or 0.0, 3))
                              for r in records if r["selected"]]
        trace["pipeline"] = self._pipeline_summary()

        thr = self.cfg["abstain_threshold"]
        if thr is not None and should_abstain(hits, thr):
            trace["verdict"] = "PASS"
            return self._finalize({"status": "abstained_low_score", "answer": "NOT_FOUND",
                                   "abstained": True, "trace": trace}, timings, t0, question, contract_id)

        g0 = clock()
        ans = self.generator.answer(question, hits)
        timings["generation_ms"] = round((clock() - g0) * 1000, 1)

        v0 = clock()
        ver = verify_answer(ans, hits)
        timings["verify_ms"] = round((clock() - v0) * 1000, 1)
        trace["verdict"] = ver.verdict

        return self._finalize({
            "status": "answered",
            "answer": ans.text,
            "abstained": ans.abstained,
            "cited_chunk_ids": ans.cited_chunk_ids,
            "quotes": ans.quotes,
            "verdict": ver.verdict,
            "passed": ver.passed,
            "usage": ans.usage,
            "evidence": self._build_evidence(ans, hits, records),
            "trace": trace,
        }, timings, t0, question, contract_id)
