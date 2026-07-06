# ClauseLens — Architecture Decision Record (ADR)

Running log of **every significant design decision**, the alternatives considered, and why
each was rejected. This directly serves the challenge's highest-weighted rubric dimension
(**Algorithm Selection & Alternative Analysis, 30%**) and feeds Section 2 of the write-up.
New decisions are appended as phases land. Decisions still open are marked and resolved by
measurement, not assertion.

**Format per entry:** Context · Decision · Alternatives rejected · Consequences · Maps to.
**Status:** ✅ Accepted · 🔄 Open (decided by ablation) · ⛔ Rejected · ♻️ Superseded

---

## D-01 — Track D (RAG), not A / B / C ✅  *(Problem Framing)*
- **Context:** Scenario 2 (enterprise documents). Exactly one track; combining tracks is explicitly penalized.
- **Decision:** RAG / LLM Knowledge Systems.
- **Alternatives rejected:**
  - **Track B (SLM fine-tuning):** contract Q&A is *knowledge retrieval*, not *skill acquisition*. Knowledge differs per document and grows over time; fine-tuning bakes stale knowledge into weights, cannot emit verbatim citations (a legal hard requirement), and needs retraining per new contract.
  - **Track A (multi-agent):** orchestration cost + failure surface with no retrieval-quality payoff for single-document clause lookup.
  - **Track C (optimization):** no natural constrained-optimization objective in document Q&A.
- **Consequences:** commit to rigorous retrieval + grounded generation + honest evaluation.
- **Maps to:** Problem Framing (25%); write-up §1.

## D-02 — Corpus = CUAD ✅  *(Problem Framing / Honest Eval)*
- **Context:** need a realistic enterprise-document corpus with a *defensible* evaluation.
- **Decision:** CUAD — 510 commercial contracts, 41 clause categories, expert-annotated answer spans (HF `theatticusproject/cuad-qa` + `theatticusproject/cuad`), CC BY 4.0.
- **Alternatives rejected:**
  - **Hand-written ~20 QA pairs (the stated minimum):** no expert labels → faithfulness becomes self-judged and circular.
  - **Synthetic contracts/QA:** needs generation + validation; real-world language gap; weaker credibility.
  - **Raw SEC EDGAR filings:** no gold answer spans → no honest retrieval labels.
- **Consequences:** character-level expert spans → *real* retrieval ground truth. The submission's core differentiator.
- **Maps to:** Honest Error Analysis (10%); Algorithm Selection (30%).

## D-03 — Eval scope = 102 official CUAD test contracts ✅  *(Honest Eval)*
- **Context:** CPU-only compute; ablations must iterate fast; retrieval is per-contract scoped.
- **Decision:** run all retrieval + e2e eval on the official 102-contract test split (4,182 QA pairs).
- **Alternatives rejected:**
  - **Full 510 contracts:** ~5× embedding cost per config for no eval benefit (retrieval is per-contract, so we only embed what we query).
  - **A random subset:** would break the official train/test split → risk of leakage and non-comparability.
- **Consequences:** fast ablation loop; still 200× the required 20 pairs. Full 510 kept only as an optional demo corpus.
- **Maps to:** Technical Execution (25%); Honest Error Analysis (10%).

## D-04 — Retrieval labels = CUAD spans → gold chunks ✅  *(Honest Eval)*
- **Context:** to compute Precision@k / Recall@k we need to know *which chunk* is correct for each query.
- **Decision:** gold chunk = any chunk overlapping an expert answer span (containment/IoU threshold); **assert every gold chunk contains its span**; unanswerables → abstention tests.
- **Alternatives rejected:**
  - **LLM-judged relevance:** circular (the thing we're evaluating grades itself) and non-reproducible.
  - **Embedding-similarity "pseudo-labels":** would bias evaluation toward the dense retriever we're testing.
- **Consequences:** metrics measured against real truth. Linchpin of the whole eval — if the span→chunk mapping is wrong, every retrieval number is wrong.
- **Maps to:** Honest Error Analysis (10%).

## D-05 — Generator LLM = Claude on AWS Bedrock ✅  *(Algorithm Selection)*
- **Context:** Track D permits external LLM APIs; user has an AWS Bedrock key; need strong grounded extraction + verbatim quoting.
- **Decision:** Claude (Haiku-class for cost) via `boto3` `bedrock-runtime` `converse` API.
- **Alternatives rejected:**
  - **OpenAI `gpt-4o-mini`:** fine, but adds a second vendor/key the user doesn't have.
  - **Free-tier (Gemini/Groq):** rate limits + less predictable grounding.
  - **Local via Ollama:** weak grounding on CPU, slow. (Kept as the named "rejected alternative" in the write-up.)
- **Consequences:** one cloud vendor; `converse` API is model-agnostic (easy to swap Claude tiers).
- **Maps to:** Algorithm Selection (30%); write-up §2, §4.

## D-06 — Vector store = FAISS flat (exact) ✅  *(Algorithm Selection)*
- **Context:** ~5–20k chunks for 102 contracts; per-contract scoped search.
- **Decision:** FAISS flat index = exact brute-force search, zero infra.
- **Alternatives rejected:**
  - **Chroma / Weaviate / pgvector / Pinecone:** managed services / ANN indexes justified at millions of vectors, not thousands. Named as the **production migration path** (→ HNSW / pgvector at scale).
  - **FAISS HNSW/IVF (approximate):** ANN trades recall for speed we don't need at this scale; exact search removes a confound from the retrieval ablation.
- **Consequences:** exact recall, reproducible, no service to run. Migration path documented for 1k-concurrency.
- **Maps to:** Algorithm Selection (30%); write-up §2, §4.

## D-07 — Ablation methodology = OFAT (one factor at a time) ✅  *(Algorithm Selection / Honest Eval)*
- **Context:** 4 axes (chunking × embeddings × retrieval × rerank) = 36 combos — too many to run and, worse, to *explain*.
- **Decision:** vary one axis at a time holding the others at a fixed baseline.
- **Alternatives rejected:**
  - **Full grid (36 cells):** confounds effects (can't attribute a delta to one variable), and expensive on CPU.
  - **Pick by intuition / literature only:** fails the rubric's "did you measure it?" probe.
- **Consequences:** every reported delta is attributable to a single change; cheaper. (Interaction effects not captured — acceptable and stated.)
- **Maps to:** Algorithm Selection (30%); Honest Error Analysis (10%).

## D-08 — Chunking strategy 🔄  *(open — decided by Phase-3 ablation)*
- **Context:** legal contracts are hierarchically numbered; clauses can straddle chunk boundaries.
- **Candidates:** recursive fixed-256 / fixed-512 @15% overlap **vs** structure-aware split on numbered clause headings (recursive fallback).
- **Alternatives to rule out via measurement:** no-overlap fixed chunks (boundary loss); whole-section chunks (dilute retrieval).
- **Resolution:** winner chosen by P@k/R@k on the eval set; boundary-loss analysis reported.
- **Maps to:** write-up Q15; Algorithm Selection (30%).

## D-09 — Embedding model 🔄  *(open — decided by Phase-3 ablation)*
- **Context:** want to know whether a free local model suffices vs a paid cloud embedding.
- **Candidates:** `BAAI/bge-small-en-v1.5` (local, CPU) **vs** a Bedrock embedding (Cohere Embed v3 / Titan v2); `bge-base` as a free local step-up.
- **Alternatives rejected up front:** OpenAI `text-embedding-3-small` (second vendor the user lacks).
- **Resolution:** compared **empirically** on P@k/R@k; report cost/quality trade-off.
- **Maps to:** write-up Q16; Algorithm Selection (30%).

## D-10 — Retrieval mode 🔄  *(open — decided by Phase-3 ablation, strong prior for hybrid)*
- **Context:** legal queries are keyword-heavy (exact terms like "indemnification", "change of control") but also paraphrased.
- **Candidates:** dense-only **vs** BM25-only **vs** hybrid dense+BM25 with **Reciprocal Rank Fusion**, ± cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`).
- **Why RRF over score-blending:** rank fusion needs no score normalization across incomparable scales.
- **Why cross-encoder over bi-encoder rerank:** joint query–doc attention → higher precision@top; accepted latency cost.
- **Resolution:** dense vs BM25 vs hybrid vs +rerank, one table, same eval set (Q17).
- **Maps to:** write-up Q17; Algorithm Selection (30%).

## D-11 — Hallucination defense = 4 layers ✅  *(Algorithm Selection / Honest Eval)*
- **Context:** legal answers must be grounded and citable; a confident wrong answer is worse than abstention.
- **Decision:** (1) grounding prompt (answer only from excerpts, cite `[chunk_id]`, emit `NOT_FOUND`); (2) **deterministic verifier** — every quoted span must be a verbatim substring of retrieved context; (3) retrieval-score threshold → abstain; (4) LLM-judge faithfulness on a sample.
- **Alternatives rejected:**
  - **Prompt-only grounding:** no detection when the model deviates.
  - **LLM-judge only:** costly, non-deterministic, can't be a hard gate.
- **Consequences:** deterministic gate catches fabricated quotes for free; judge used only for the reported metric.
- **Maps to:** write-up Q18; Algorithm Selection (30%).

## D-12 — Corrective RAG (CRAG): partially adopt, reject web fallback 🔄  *(Algorithm Selection)*
- **Context:** CRAG (Yan et al. 2024) assesses retrieval quality and corrects; its "incorrect" branch does a **web search**.
- **Decision:** adopt CRAG's retrieval-assessment + correction *routing*; **replace web search with within-corpus query reformulation + abstention**. Kept only if the Phase-5 A/B reduces the retrieval-miss bucket.
- **Alternatives rejected:**
  - **CRAG as published (web fallback):** any answer sourced outside the executed contract is worse than abstaining — breaks grounding + citability.
- **Consequences:** a corpus-scoped corrective loop; measured, not assumed.
- **Maps to:** Algorithm Selection (30%); write-up §2.

## D-13 — Cache-Augmented Generation (CAG): reject as primary, adopt as optimization 🔄  *(Algorithm Selection)*
- **Context:** CAG (Chan et al.) preloads the whole document into context (KV cache) and skips retrieval; tempting because our queries are per-contract.
- **Decision:** **reject as the primary system**; **adopt** its idea as (a) session-level prompt-caching of the contract on Bedrock (cost/latency win) and (b) a RAG-vs-CAG ablation on short contracts (Phase 5).
- **Alternatives rejected:**
  - **CAG as primary:** (1) off-track — Track D mandates a retrieval pipeline; (2) tail contracts exceed useful context + ~15× per-query token cost; (3) blurs verbatim-citation grounding.
- **Note:** true KV-cache CAG needs a self-hosted model; via Bedrock it's *prompt caching*, not KV reuse.
- **Maps to:** Algorithm Selection (30%); write-up §2, §4.

## D-14 — Self-RAG ⛔  *(Algorithm Selection)*
- **Decision:** rejected. Reflection-token self-critique requires **fine-tuning** a model — wrong fit for a no-fine-tune 48h RAG build (and closer to Track B). Named in the write-up as a considered alternative.

## D-15 — Adaptive-RAG ⛔  *(Algorithm Selection)*
- **Decision:** rejected. Routes by query complexity; overkill when all CUAD queries are single-hop clause lookups. Named as considered.

## D-16 — GraphRAG ⛔  *(Algorithm Selection)*
- **Decision:** rejected. Knowledge-graph construction is heavy and shines for multi-hop cross-document synthesis, not single-clause lookup within one contract. Named as **future work** (cross-contract obligations analysis).

## D-17 — Isolated venv + pinned requirements ✅  *(Technical Execution)*
- **Context:** rubric grades "reproducible from scratch"; global Python has 217 unrelated packages.
- **Decision:** per-project `.venv`, pinned `requirements.txt`, stated Python 3.13.2; venv gitignored (build output, not source).
- **Alternatives rejected:** global installs (non-portable, entangled); conda (heavier, unnecessary here).
- **Maps to:** Technical Execution (25%).

## D-18 — Phase-by-phase build with per-phase commits ✅  *(Communication / Technical Execution)*
- **Context:** "a single bulk commit is a red flag"; commit history is reviewed.
- **Decision:** build one phase at a time; each phase = decision-labeled commit(s) + push; concept brief + interview Q&A per phase.
- **Alternatives rejected:** one big commit at the end (opaque, penalized).
- **Maps to:** Communication (10%); Technical Execution (25%).

---

*Open decisions (D-08, D-09, D-10, D-12, D-13) are resolved with numbers as Phases 3 and 5 land; this file is updated in place with the empirical winner and the measured deltas.*
