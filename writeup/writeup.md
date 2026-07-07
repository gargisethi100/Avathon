# ClauseLens — RAG for Contract Intelligence

**Gargi Sethi**  ·  **Track D — RAG / LLM Knowledge Systems**  ·  Scenario 2 (Gen AI for Enterprise Documents)
*(name inferred from repo — please verify before submission)*

## 1 · Problem & Domain

Legal teams drown in contracts and need to answer natural-language questions about a *specific* document — "what is the termination-for-convenience notice period?" — with **citable, verifiable** answers. This is a **knowledge-retrieval** problem, not a skill-acquisition one: the knowledge (clause text) differs per document and grows over time. That is exactly why **RAG (Track D), not fine-tuning (Track B)** — fine-tuning bakes stale knowledge into weights, cannot emit verbatim citations (a legal hard requirement), and needs retraining per new contract; **not agents (Track A)** — orchestration cost with no retrieval-quality payoff for single-document clause lookup.

**Corpus: CUAD** (510 commercial contracts, 41 clause categories, expert-annotated answer spans; CC BY 4.0). The spans are the differentiator: they give *real* retrieval ground truth, so we **measure** retrieval quality honestly rather than self-judging it. Test partition: 102 contracts, 4,182 QA pairs (70% unanswerable — most clause categories are absent from any given contract, which we treat as a large, honest abstention test).

## 2 · Approach & Algorithm Decisions

Every major choice was **decided by a measured ablation**, not asserted — and several priors were *refuted* by the data.

- **Chunking (Q15) → recursive-512.** Compared recursive-256/512 vs structure-aware (clause-heading) splitting. 512 beat 256 decisively (R@10 0.75 vs 0.67 — long legal clauses fragment at 256). Against our prior, **structure-aware did *not* beat recursive** (near-tie), so we ship the simpler splitter. 15% overlap + verified **100% gold-span coverage** at every size → no answer is lost at a chunk boundary.
- **Embeddings (Q16) → bge-small (local, free).** Empirically tied with the 3×-larger bge-base (within ~1 pt R@10) → a free local model suffices; a cost/latency win at no accuracy cost. OpenAI rejected (second vendor); Bedrock-Cohere kept as an optional cloud cross-check.
- **Retrieval + fusion (Q17) → hybrid dense+BM25 with Reciprocal Rank Fusion.** Measured dense / BM25 / hybrid / +rerank. **BM25 beat dense** (R@5 0.54 vs 0.47 — legal queries are keyword-heavy); hybrid-RRF edged out BM25. **A MS-MARCO cross-encoder rerank *hurt*** (0.55→0.48): a web-trained reranker mis-transfers to contract language — so we ship hybrid-RRF *without* it and name a legal-domain reranker as future work. RRF over score-blending (rank fusion needs no normalization across incomparable cosine/BM25 scales). **FAISS flat** (exact search; ANN needless at ~thousands of vectors; pgvector/HNSW named as the production path).
- **Generation → Claude Haiku 4.5 on AWS Bedrock.** Grounding prompt: answer *only* from excerpts, quote verbatim, cite `[n]`, else output `NOT_FOUND`.
- **Hallucination defense (Q18), 4 layers:** grounding prompt → **deterministic verbatim-quote verifier** (every quoted span must be a substring of retrieved context — catches fabrication for free, no LLM cost) → retrieval-score-threshold abstention → LLM-judge on a sample. *Finding:* the verifier must normalize whitespace **and punctuation spacing**, because LLMs silently tidy quotes (source `"China ,otherwise"` → quote `"China, otherwise"`); without it, faithfulness is under-reported.
- **Query gate:** a light Claude classifier routes each query → IN_SCOPE / OUT_OF_SCOPE / AMBIGUOUS, so out-of-scope questions are declined gracefully instead of forced through retrieval.

**Considered and rejected (named):** **Corrective RAG** — adopt its assess-and-correct routing, reject its web-search fallback (external content ≠ the executed contract); **Cache-Augmented Generation** — rejected as primary (off-track, tail contracts exceed context ~75k tokens, blurs citation), adopted as a session prompt-cache; **Self-RAG / Adaptive-RAG / GraphRAG** — named, rejected as ill-fitting a no-fine-tune single-hop task; **RAGAS** — not primary, because we have real gold labels and *measure* retrieval rather than LLM-judge it (RAGAS kept as an optional cross-check).

## 3 · Results & Error Analysis  *(held-out CUAD test)*

**Retrieval OFAT** (244-query, 20-contract stratified subset — CPU tractability; relative ranking is what the ablation decides):

| axis | winner | key numbers |
|---|---|---|
| retrieval mode (Q17) | **hybrid-RRF** | dense R@5 .47 · BM25 .54 · **hybrid .55** · +rerank .48 (rerank *hurt*) |
| chunking (Q15) | **recursive-512** | R@10: 256 = .665, **512 = .753**; structure-aware tied, not better |
| embedding (Q16) | **bge-small** | bge-small R@10 .665 ≈ bge-base .656 (no lift) |

**End-to-end** (balanced 300-pair sample; recursive-512 + bge-small + hybrid-RRF; Claude Haiku 4.5): answer token-F1 **0.58 when it answers** (0.44 incl. abstentions) · abstention **precision 0.73 / recall 0.83** · strict verbatim-verifier faithfulness **0.53**. Error taxonomy (300): clean-answer 48 · clean-abstain 125 · unfaithful-quote 46 · wrongful-abstention 46 · wrongful-answer 25 · correct-context-wrong-answer 10.

Three honest findings (this is where the system actually breaks):
1. **Faithfulness 0.53 is a strict lower bound, not "hallucinates half the time."** The gate flags an answer if *any* quoted span is non-verbatim; verbose answers pair a **correct** primary quote (F1 up to 1.0) with paraphrased "key-point" quotes that fail. Verifying only the primary citation, or a prompt that suppresses embellishment, would raise it — a clear next step. The strictness is deliberate: it's a hard safety gate, not the reported semantic faithfulness (an LLM judge would score higher).
2. **Wrongful abstention (46) is not purely a retrieval miss.** A large slice is the "Document Name" category, where the gold answer is the contract *title* and the model — though the title is retrieved — declines to treat "the contract is called X" as the answer. So it is part retrieval-recall, part question-interpretation.
3. **Abstention recall 0.83:** 25 unanswerable questions were answered (the model over-explains a *related* clause instead of a clean NOT_FOUND); the deterministic verifier flags most of these as UNGROUNDED — a partial safety net even when generation over-reaches.

*Caveat:* the retrieval ablation ran on a 20-contract subset (CPU tractability); the e2e eval used the full balanced 300-pair sample.

## 4 · Production & Limitations

**Production (1,000 concurrent queries):** bottlenecks are (a) embedding/rerank CPU → GPU batch server or lighter model; (b) LLM API rate limits → response caching + request queue; (c) FAISS flat → HNSW/pgvector. **Freshness:** per-contract incremental add (no corpus re-embed); embedding-model version change is the real re-embed trigger. **Cost:** Haiku 4.5 + a 300-pair eval is a few cents; Bedrock prompt-caching amortizes repeated per-contract context across a session.

**Limitations:** (1) **tables/exhibits** — text-only chunking loses tabular layout; (2) **multi-span answers** (mean 2.1, max 19 spans/question) stress single-chunk retrieval; (3) the single biggest lever is **retrieval recall** — a legal-domain reranker or a fine-tuned embedder is the clear next step, and would directly shrink the wrongful-abstention bucket.
