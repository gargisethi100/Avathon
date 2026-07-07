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

## D-08 — Chunking: recursive-512 ✅  *(resolved by ablation — Q15)*
- **Measured** (244-query subset, hybrid, bge-small): recursive-256 R@10 0.665 / nDCG 0.488; **recursive-512 R@10 0.753 / nDCG 0.522**; structure_aware-256 0.669 / 0.482; structure_aware-512 0.744 / 0.520.
- **Winner: recursive-512** — bigger chunks win decisively (256→512 lifts R@10 ~+9 pts; long legal clauses fragment at 256). **Counter to our prior, structure-aware did NOT beat recursive** (near-tie at 512) — prior corrected by measurement, and recursive is simpler (no heading parser).
- **Boundary loss (Q15):** mitigated by 15% overlap; 100% gold-span coverage at every size (`chunk_coverage.md`) → no answer is lost to a boundary.
- **Maps to:** write-up Q15; Algorithm Selection (30%).

## D-09 — Embedding: bge-small (bge-base gives no lift) ✅  *(resolved by ablation — Q16)*
- **Measured** (recursive-256, hybrid): bge-small R@5 0.546 / R@10 0.665 / nDCG 0.488; bge-base R@5 0.543 / R@10 0.656 / nDCG 0.496 — **within ~1 pt, essentially tied**.
- **Winner: bge-small** — the 3×-larger, ~3×-slower-to-embed bge-base gives no meaningful improvement. Answers Q16's "does a free local model suffice?" → **yes** — a cost/latency win at no accuracy cost.
- **Bedrock-Cohere** embedding kept as an optional cloud cross-check (creds available); not needed to answer Q16. OpenAI `text-embedding-3-small` rejected up front (second vendor).
- **Maps to:** write-up Q16; Algorithm Selection (30%).

## D-10 — Retrieval mode: hybrid-RRF; rerank HURT ✅  *(resolved by ablation — Q17)*
- **Measured** (recursive-256, bge-small): dense R@5 0.469 / nDCG 0.437; **bm25 R@5 0.543 / nDCG 0.481**; **hybrid-RRF R@5 0.546 / nDCG 0.488**; hybrid+rerank R@5 0.482 / nDCG 0.452.
- **Winner: hybrid-RRF** (RRF fuses by rank → no score normalization across the incomparable cosine/BM25 scales). Two notable findings:
  1. **BM25 > dense** on this corpus — legal queries are keyword-heavy; sparse retrieval vindicated. Hybrid edges out BM25.
  2. **The cross-encoder rerank HURT** (R@5 0.546 → 0.482): `ms-marco-MiniLM-L-6-v2` is web(MS-MARCO)-trained and mis-transfers to contract text. So we **ship hybrid-RRF WITHOUT the generic reranker** — a legal-domain reranker (`bge-reranker-base`) is named as future work, not assumed to help.
- **Maps to:** write-up Q17; Algorithm Selection (30%); Honest Error Analysis (10%).

## D-11 — Hallucination defense = 4 layers ✅  *(Algorithm Selection / Honest Eval)*
- **Context:** legal answers must be grounded and citable; a confident wrong answer is worse than abstention.
- **Decision:** (1) grounding prompt (answer only from excerpts, cite `[chunk_id]`, emit `NOT_FOUND`); (2) **deterministic verifier** — every quoted span must be a verbatim substring of retrieved context; (3) retrieval-score threshold → abstain; (4) LLM-judge faithfulness on a sample.
- **Alternatives rejected:**
  - **Prompt-only grounding:** no detection when the model deviates.
  - **LLM-judge only:** costly, non-deterministic, can't be a hard gate.
- **Consequences:** deterministic gate catches fabricated quotes for free; judge used only for the reported metric.
- **Phase-4 finding:** the verifier must normalize whitespace **and spacing around punctuation** — LLMs silently tidy quotes (source `"China ,otherwise"` → quote `"China, otherwise"`), so a whitespace-only check false-flags grounded answers as UNGROUNDED and would under-report faithfulness. Fixed; validated end-to-end (answerable → PASS, fabricated → UNGROUNDED).
- **Phase-5 finding:** strict "*all* quotes must be verbatim" gives a **conservative** faithfulness of 0.53 on the 300-pair sample — verbose answers pair a correct primary quote (F1↑) with paraphrased "key-point" quotes that fail. It's a hard *lower bound*; verifying only the primary citation (or suppressing embellishment in the prompt) would raise it. This is why the layered design also keeps an LLM judge for the *reported* semantic-faithfulness rate.
- **Maps to:** write-up Q18; Algorithm Selection (30%); Honest Error Analysis (10%).

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
- **Evidence (Phase-1 EDA):** contract length median ≈25.7k chars (~6k tokens — fits context) but max ≈300.8k chars (~75k tokens) — the tail confirms the context/cost failure mode.
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

## D-19 — CUAD via HF auto-converted Parquet, not the loading script ✅  *(Technical Execution)*
- **Context:** `datasets>=5` removed support for dataset *loading scripts*; `theatticusproject/cuad-qa` ships only `cuad-qa.py` (no data files on the default branch).
- **Decision:** pull the `refs/convert/parquet` export directly via `huggingface_hub.hf_hub_download` + pandas.
- **Alternatives rejected:**
  - **Downgrade `datasets` to a script-supporting version:** un-pins the env, risks Py3.13 incompatibility, less reproducible.
  - **Vendor raw `CUAD_v1.json` from Zenodo/GitHub:** heavier, and we'd re-derive the official train/test split ourselves (divergence risk).
- **Consequences:** script-free and reproducible; `data/download_cuad.py` re-validates the split (102 contracts / 4,182 QA) on every run.

## D-20 — TLS via `truststore` (OS trust store), not certifi ✅  *(Technical Execution)*
- **Context:** dev machine sits behind a corporate SSL-inspection proxy; its root CA is only in the Windows store, so `certifi` fails `CERTIFICATE_VERIFY_FAILED` on every HF request.
- **Decision:** `truststore.inject_into_ssl()` at the top of networked scripts → Python uses the OS trust store (trusts the corporate CA). Guarded in try/except → no-op off-corporate.
- **Alternatives rejected:**
  - **Disable SSL verification:** insecure; unacceptable even in a dev tool.
  - **Export the corporate CA to a PEM + `SSL_CERT_FILE`:** brittle, machine-specific, manual steps to reproduce.
- **Consequences:** all HF downloads (data + `bge`/cross-encoder models) work securely on the corporate network.

## D-21 — Balanced e2e eval sample (150/150); true distribution reported separately ✅  *(Honest Eval)*
- **Context:** CUAD test is **70.3% unanswerable**; a natural-distribution ~300 sample leaves too few answerable cases for stable answer-quality metrics.
- **Decision:** seeded, category-stratified draw of **150 answerable + 150 unanswerable** (covers 41/41 categories, 96/102 contracts). Report answer-F1 on the answerable half and abstention precision/recall on the unanswerable half, **and** report headline metrics on the true 70/30 distribution.
- **Alternatives rejected:**
  - **Mirror true 70% unanswerable:** ~90 answerable → noisy answer-F1.
  - **Answer-heavy sample:** under-tests abstention — the safety-critical behavior in legal.
- **Consequences:** statistical power on both behaviors; distribution honesty preserved via dual reporting.
- **Honest-eval caveat:** some categories (e.g. *Price Restrictions*) have **0 positives** in the test split → retrieval can't be measured there; excluded from per-category retrieval stats and flagged.
- **Maps to:** Honest Error Analysis (10%); write-up §3.

## D-22 — Chunk sizing decoupled from the embedder; query/passage asymmetry ✅  *(Technical Execution / Honest Eval)*
- **Context:** the embedding model is a *measured* axis (D-09) — chunk boundaries must not depend on which embedder we test, or the comparison is confounded.
- **Decision:** size chunks in **nominal chars (tokens × 4)**, independent of any model tokenizer, so **every backend embeds the identical chunks**. Retrieval encoding respects each model's query/passage asymmetry (bge query prefix; Cohere `input_type`).
- **Alternatives rejected:**
  - **Size chunks with the embedder's own tokenizer:** boundaries shift per embedder → the embedding ablation confounds "better model" with "different chunks".
  - **Symmetric query/passage encoding:** ignores how bge/Cohere were trained for retrieval → weaker recall.
- **Verified (Phase 2):** offset round-trip 0 failures; **gold-span coverage = 100%** for all 4 chunk configs → retrieval recall ceiling is 100% (any miss in Phase 3 is a retriever failure, not chunking). See `results/chunk_coverage.md`.
- **Maps to:** write-up Q15/Q16; Technical Execution (25%).

## D-23 — Real-label metrics as primary eval, not RAGAS ✅  *(Honest Eval / Algorithm Selection)*
- **Context:** RAGAS is the standard RAG-eval framework (faithfulness, context precision/recall, answer relevancy), but its metrics are largely **LLM-judged / reference-free** — designed for setups that LACK ground truth.
- **Decision:** primary eval = metrics against **CUAD's real gold spans** — retrieval P@k/R@k/MRR/nDCG (real labels), deterministic verbatim-quote faithfulness, token-F1 vs gold. Keep RAGAS as an **optional secondary cross-check** on the e2e sample (faithfulness / answer-relevancy, on Bedrock), time permitting.
- **Alternatives rejected (as primary):**
  - **RAGAS context precision/recall (LLM-judged):** strictly weaker than real-label retrieval metrics — estimating with an LLM what we can measure injects noise + leakage risk.
  - **RAGAS faithfulness (LLM-judge only):** non-deterministic and costly; our deterministic verbatim gate is exact and free, with the LLM judge kept as a sampled add-on.
- **Consequences:** fully transparent, line-by-line-defensible numbers; RAGAS named as a considered framework and available as a robustness cross-check.
- **Maps to:** Honest Error Analysis (10%); Algorithm Selection (30%); write-up §3.

## D-24 — Conversational memory: condense-question + bounded summarized history ✅  *(query handling / production)*
- **Context:** the pipeline is stateless; referential follow-ups ("and its notice period?") break **retrieval** (unresolved pronoun) and trip the gate as AMBIGUOUS.
- **Decision:** **query contextualization (condense-question)** — an LLM rewrites each follow-up into a standalone question from the running summary + recent turns, then the *unchanged* gate → retrieve → generate → verify pipeline runs. Memory = **last-5 turns verbatim + a running summary of older turns**, **contract-scoped** (reset on switch). In `src/conversation.py` (`ClauseLensSession`), reusing the Bedrock client.
- **Alternatives rejected:**
  - **Stuff full history into the generator only:** retrieval still breaks on the pronoun — retrieval, not generation, is the failure point.
  - **Unbounded history** (token/latency blowup) / **summary-only** (loses recent specifics) — the hybrid bounds cost while keeping recent turns exact.
- **Grounding preserved:** memory resolves references only, never adds knowledge; answers stay retrieved-grounded, cited, and verifier-checked against *this* contract. Validated: "and its notice period?" → "…notice period … for termination …", and later turns auto-inject the parties' names.
- **Maps to:** query handling (strong-submission item); Algorithm Selection (30%); production.

---

*Decisions D-08/D-09/D-10 were resolved by the Phase-3 ablation; D-12/D-13 (CRAG/CAG) remain future-work options. This file is updated in place as new decisions land.*
