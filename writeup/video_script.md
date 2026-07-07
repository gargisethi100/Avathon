# ClauseLens — 5-Minute Walkthrough Script

Screen-record the demo + a couple of tables. Focus on **decisions and reasoning**, not code.
Target 5:00. Timestamps are guides.

---

## 0:00–1:00 · Problem & approach choice
- "ClauseLens answers questions about a *specific* commercial contract with **verbatim, verifiable citations**. Legal teams can't act on an answer they can't trace to the clause."
- "This is a **knowledge-retrieval** problem, not skill-acquisition — so **RAG (Track D)**, not fine-tuning: fine-tuning bakes stale knowledge into weights, can't cite, and needs retraining per new contract."
- "Corpus is **CUAD** — 510 real contracts with **expert-annotated answer spans**. That's the key: I have *real* retrieval ground truth, so I **measure** quality instead of eyeballing it."

## 1:00–3:00 · Key algorithm decisions & what I ruled out
> Theme: *every choice was decided by measurement — and three of my priors were refuted.*
- **Retrieval (the headline):** "I compared dense, BM25, hybrid, and +rerank. Two surprises: **BM25 beat dense** — legal queries are keyword-heavy — and a **cross-encoder reranker actually *hurt*** results. It's MS-MARCO/web-trained and mis-transfers to contract language. So I ship **hybrid-RRF without the reranker**. If I'd assumed 'reranking helps,' I'd have shipped a worse system."
- **Chunking:** "512-token chunks beat 256 — long clauses fragment at 256. I *built* a structure-aware chunker expecting it to win on numbered contracts; it **tied**, so I ship the simpler recursive splitter."
- **Embeddings:** "bge-small **tied** the 3×-bigger bge-base — a free local model suffices. I have the numbers, not a vibe."
- **Hallucination defense:** "Four layers, and the cheap one is the best: a **deterministic verifier** — every quoted span must be a verbatim substring of the retrieved context. Catches fabrication for free. (I also found it must normalize punctuation, because the model silently tidies quotes.)"
- "Alternatives I named and rejected: Corrective RAG's web fallback — unsafe, external text isn't the contract; CAG — off-track and blows the context budget on long contracts; RAGAS — I have real labels, so I measure rather than LLM-judge."

## 3:00–4:30 · Results & what didn't work
- Show `results/retrieval_metrics.md`: "recursive-512 + bge-small + hybrid-RRF wins."
- Show the demo: an **answerable** query → cited, quoted, green faithfulness badge; an **unanswerable** clause → **NOT_FOUND** (it abstains); an **out-of-scope** query → the gate declines before retrieval.
- Honest failure: "End-to-end, **abstention recall is ~1.0** — it never fabricates an answer for an absent clause. The cost is **over-abstention**: my dominant error is **wrongful abstention driven by retrieval misses** — when the gold chunk is outside top-k, the model correctly says NOT_FOUND. So the ceiling is retrieval **recall**, not generation."
- Caveat stated plainly: "Ablation ran on a 20-contract subset for CPU tractability; the relative ranking is what it decides."

## 4:30–5:00 · What I'd do with more time
- "A **legal-domain reranker** (bge-reranker or fine-tuned) — directly attacks the wrongful-abstention bucket."
- "Handle **tables/exhibits** (layout-aware chunking) and **multi-span** answers."
- "Full 102-contract e2e eval + a **RAGAS cross-check** to triangulate faithfulness."
- Close: "Every number here is one I can defend — that was the goal."
