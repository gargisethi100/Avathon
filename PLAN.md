# ClauseLens — Phased Build + Interview-Readiness Plan (Avathon Track D / Scenario 2)

## Context

This is an Avathon AI hiring challenge submission. The scoring rubric weights **reasoning over accuracy**: Algorithm Selection & Alternatives (30%) + Problem Framing (25%) = 55% of the score rewards *knowing why*, not raw metrics. Shortlisted candidates also face a 60-min technical interview (walk through a design decision, live extension, system design at scale).

Therefore the real deliverable is **the candidate's deep understanding of every design choice** — not just working code. This plan is built to teach as it builds: each phase pairs a shippable artifact with a **concept brief** (the idea + why it beat alternatives) and **interview Q&A** (likely questions with model answers). We build **one phase at a time**, and each phase is committed and pushed to git before the next begins.

Chosen combination: **Scenario 2 (Gen AI for Enterprise Documents) × Track D (RAG)**. Product framing: *ClauseLens* — a legal-team assistant that answers natural-language questions about a contract corpus with verbatim citations, faithfulness verification, and graceful abstention.

Current state: `c:\Users\gargi.sethi\Desktop\RAG` is an empty git repo (remote `Origin` → `github.com/gargisethi100/Avathon`, a 2-line README not yet checked out locally). Everything is built from scratch, **in-place in this directory**. Python 3.13.2 already has `faiss-cpu`, `sentence-transformers`, `torch`, `transformers`, `streamlit`.

## Locked decisions (from user)

| Decision | Choice | Consequence |
|---|---|---|
| Build location | **In-place** in `c:\Users\gargi.sethi\Desktop\RAG` (existing `Avathon` repo); `PLAN.md` copied to repo root | Everything lives in the project directory + git history |
| Generator LLM | **Claude on AWS Bedrock** (`boto3` `bedrock-runtime` `converse` API, Haiku-class for cost) | No OpenAI dependency; confirm model access + region at build time |
| Embedding comparison (Q16) | **bge-small (local) vs a Bedrock embedding** (Cohere Embed v3 / Titan v2) + bge-base local as free fallback | "Does a free local model suffice?" story using existing key |
| Corpus scope | **102 official CUAD test contracts** | All eval (retrieval + e2e) runs on this; 4,182 QA pairs; fast CPU loop. Full 510 only for a bigger demo corpus if time allows |
| Learning format | **Concept brief + interview Q&A per phase** | Each phase ends with a short written explainer + practice questions |
| Workflow | **Phase by phase; commit + push each phase to `Origin/main`** | Phases are the steps. No bulk commits (rubric red flag). Start with the `.venv` |

## Refined architecture

```
102 test contracts ── Ingestion: parse → chunk → embed → index (FAISS flat + BM25)
User query ── Query gate (out-of-scope / ambiguity check)
           ── Hybrid retrieval (dense + BM25, RRF fusion, contract_id filter)
           ── Cross-encoder re-ranking → top-k context
           ── [optional] CRAG-adapted corrective loop (reformulate-in-corpus OR abstain)
           ── Generator: Claude on Bedrock (grounded prompt: quote → cite chunk IDs → or abstain)
           ── Faithfulness verifier (deterministic verbatim-quote check + LLM-judge sample)
           ── Answer + citations + confidence badge   |   graceful abstention
```

**Two methodology refinements over a naive build (both are interview-strengtheners):**

1. **OFAT ablations, not a full 36-cell grid.** Change **one axis at a time**, holding the others at a fixed baseline. Cheaper on CPU, and *isolates each variable's effect* so every number is attributable. A full grid confounds effects and can't be defended cleanly in an interview.
2. **Retrieval is per-contract scoped**, so we only embed the 102 contracts we query — the ablation loop is fast. Corpus-wide search is a demo mode, not the eval mode.

## Working agreement (how we proceed)

- **Phases are the steps.** We do one phase at a time. Each phase = (1) build the component(s), (2) **concept brief + interview Q&A**, (3) **`git add` → commit (decision-focused message) → `git push Origin main`**. I pause after each phase for your questions before starting the next.
- **First action of Phase 0 is creating the project `.venv`** — the concrete first step you asked for.
- No bulk commits — the commit history itself is graded and should read like an iterative build.
- Reproducibility is graded: `.venv`, pinned `requirements.txt` (incl. Python version), seeded runs, README that reproduces results top-to-bottom.

## Repository structure (target)

```
├── README.md                # problem, data sourcing, setup, full reproduction guide
├── PLAN.md                  # this plan, at repo root
├── requirements.txt         # pinned + Python 3.13
├── .gitignore               # exclude challenge PDF ("Not for Distribution"), data/, indexes/, .venv/
├── data/
│   ├── download_cuad.py     # fetch cuad-qa + raw contracts (HF datasets)
│   ├── build_eval_set.py    # gold-chunk mapping, stratified e2e sample, OOS queries
│   └── eda.ipynb            # corpus stats for write-up (incl. per-contract token distribution → CAG boundary)
├── src/
│   ├── chunking.py          # recursive-256/512 + structure-aware (pluggable); tracks char offsets
│   ├── embeddings.py        # bge-small/bge-base (local) + Bedrock embedding backend (pluggable)
│   ├── indexing.py          # FAISS flat + BM25 build, incremental add
│   ├── retrieval.py         # dense / sparse / hybrid-RRF / +rerank / [optional] corrective loop
│   ├── generation.py        # Bedrock Claude client, grounded prompt, citation parsing, abstention
│   ├── verifier.py          # verbatim-quote check, score thresholds, faithfulness judge
│   └── pipeline.py          # config-driven orchestration + trace logging
├── eval/
│   ├── retrieval_eval.py    # P@k, R@k, MRR, nDCG across OFAT ablations
│   ├── e2e_eval.py          # token-F1/Jaccard vs gold spans, faithfulness, abstention P/R, RAG-vs-CAG
│   └── error_analysis.py    # failure buckets with examples
├── configs/                 # YAML per ablation cell (seeded)
├── results/                 # metrics tables, plots, annotated Q&A traces
├── app.py                   # Streamlit demo
└── writeup/                 # 1–2 page PDF source
```

---

## Phased roadmap (each phase = one committed + pushed step)

Each phase lists: **Goal · Build · Key files · Concept brief topics · Interview Q&A anchors.**

### Phase 0 — Scaffold & environment (~0.5h)
- **Goal:** reproducible foundation + clean git history.
- **Build (in order):** **(1) create the project `.venv`** (`python -m venv .venv`, activate, upgrade pip) → **(2)** reconcile local `main` with `Origin/main` so the existing README lands → **(3)** pinned `requirements.txt` (add `datasets`, `rank-bm25`, `boto3`) installed into `.venv` → **(4)** repo skeleton + `PLAN.md` at root + `.gitignore` (PDF, data/, indexes/, .venv/) + README stub. Commit + push.
- **Concept brief:** why pinned deps + seeds + venv = the "reproducible from scratch" the rubric grades; why incremental commits matter (bulk commit = red flag).
- **Interview Q&A:** "How do you make an ML result reproducible?" · "Why a venv, not global installs?"

### Phase 1 — Data + gold labels *(the honest-eval foundation)* (~3h)
- **Goal:** turn CUAD's expert span annotations into ground-truth retrieval labels. **This is the linchpin — if the span→chunk mapping is wrong, every retrieval metric is wrong.**
- **Build:** `download_cuad.py` (theatticusproject/cuad-qa + raw contracts); `eda.ipynb` (lengths, structure, 41-category skew, **per-contract token distribution**); `build_eval_set.py` — gold chunks = chunks overlapping the annotated answer span (containment/IoU threshold); stratified ~300-pair e2e sample (~30% unanswerable); hand-written OOS + ambiguous queries. **Assert every gold chunk contains its span; log label counts.**
- **Key files:** `data/download_cuad.py`, `data/build_eval_set.py`, `data/eda.ipynb`.
- **Concept brief:** SQuAD-2.0 format; character-offset span→chunk mapping; containment vs IoU; unanswerables → abstention tests; per-contract scoping; why real labels beat self-generated "faithfulness" judgments.
- **Interview Q&A:** "How do you know your eval labels are correct?" · "How do you evaluate what the system *shouldn't* answer?"

### Phase 2 — Ingestion: chunking + embeddings + indexing (~3h)
- **Goal:** pluggable ingestion so chunking/embedding become *measured* axes, not assumptions.
- **Build:** `chunking.py` (recursive fixed-256 / fixed-512 @15% overlap + structure-aware on numbered clause headings, recursive fallback; **preserves char offsets**); `embeddings.py` (bge-small + bge-base local; Bedrock embedding backend); `indexing.py` (FAISS flat + BM25 via `rank_bm25`; incremental add — FAISS IDMap append + BM25 corpus append).
- **Key files:** `src/chunking.py`, `src/embeddings.py`, `src/indexing.py`.
- **Concept brief:** chunking tradeoffs (boundary loss on straddling clauses; overlap + structure-awareness as mitigation); dense (bi-encoder) vs sparse (BM25); why **FAISS flat = exact search** (no ANN at ~5–20k chunks, zero infra); Chroma/Weaviate/pgvector named as production migration path.
- **Interview Q&A:** "Why this chunk size/overlap? What's lost at boundaries (Q15)?" · "Why FAISS flat over a managed vector DB?"

### Phase 3 — Retrieval modes + ablations (~5h)
- **Goal:** decide retrieval mode by measurement; produce the headline ablation tables.
- **Build:** `retrieval.py` (dense / BM25-only / hybrid-RRF / hybrid+cross-encoder rerank, contract_id filter); `retrieval_eval.py` (P@5, R@5, R@10, MRR, nDCG@10). Run **OFAT** sweeps: chunking axis, embedding axis, retrieval-mode axis, rerank on/off — each holding others at baseline. Config-driven + seeded.
- **Key files:** `src/retrieval.py`, `eval/retrieval_eval.py`, `configs/*.yaml`.
- **Concept brief:** RRF math; why hybrid helps legal keyword-heavy queries ("indemnification", "change of control") where dense alone misses exact terms; cross-encoder (joint query-doc attention) vs bi-encoder — accuracy vs latency; OFAT methodology.
- **Interview Q&A:** "How does fusion/reranking beat a single retriever — show it (Q17)?" (answer *with your numbers*) · "Why did BM25 matter for this corpus specifically?"

### Phase 4 — Generation + verification + query gate (~3h)
- **Goal:** grounded answers with layered hallucination defense (write-up Q18).
- **Build:** `generation.py` (Bedrock Claude `converse`; prompt contract: answer ONLY from excerpts, quote verbatim, cite `[chunk_id]`, output `NOT_FOUND` when absent; citation parsing); `verifier.py` (deterministic check — every quoted span must be a verbatim substring of retrieved context, else flag/refuse; retrieval-score threshold → abstain; LLM-judge faithfulness on a sample); query gate (embedding-similarity threshold + light LLM check for OOS/ambiguity).
- **Optional measured stage — CRAG-adapted corrective loop:** relevance-gate the reranked context → if low-confidence, one **within-corpus** query reformulation + re-retrieve → else abstain. **No web fallback** (any answer sourced outside the executed contract is worse than abstaining). Kept only if the Phase-5 A/B shows it reduces the retrieval-miss bucket.
- **Key files:** `src/generation.py`, `src/verifier.py`, `src/pipeline.py`.
- **Concept brief:** the 4-layer defense (grounding prompt → deterministic verbatim verifier → score-threshold abstention → judge-sampled metric); deterministic vs LLM-judge verification; why abstention beats guessing in legal; how the corrective loop adapts CRAG safely.
- **Interview Q&A:** "How do you prevent hallucination and *detect* it (Q18)?" · "How do you handle out-of-scope / ambiguous queries?" · "Would Corrective RAG help here?"

### Phase 5 — End-to-end eval + honest error analysis (~4h)
- **Goal:** end-to-end numbers on the winning config + a failure taxonomy (the 10% Error Analysis dimension).
- **Build:** `e2e_eval.py` on the ~300-pair sample — token-F1/Jaccard vs gold spans (answerables), abstention precision/recall (unanswerables), faithfulness rate (verifier + judge). **RAG-vs-CAG ablation** on short contracts (retrieve top-k vs whole-contract-in-context) on F1 + cost/latency → shows *when retrieval earns its cost*. Optional corrective-loop A/B. `error_analysis.py` buckets: retrieval miss / partial retrieval / correct-context-wrong-answer / unfaithful quote / wrongful abstention — with real examples.
- **Key files:** `eval/e2e_eval.py`, `eval/error_analysis.py`, `results/`.
- **Concept brief:** token-F1/Jaccard for span answers; abstention P/R vs accuracy; faithfulness measurement; honest error taxonomy; RAG-vs-CAG cost/precision tradeoff.
- **Interview Q&A:** "Where does your system fail and why?" · "How do you measure faithfulness rigorously?" · "When is retrieval even necessary vs CAG?"

### Phase 6 — Demo + representative traces (~2h)
- **Goal:** a demo that shows *reasoning*, not just output.
- **Build:** `app.py` (Streamlit: retrieved chunks + scores, citations, confidence badge; live abstention + OOS handling); ≥5 annotated Q&A traces in `results/` (incl. 1 unanswerable → abstains, 1 OOS → graceful decline).
- **Concept brief:** what a strong demo surfaces (retrieval transparency, citation verifiability); prep for the interview's "live extension" segment.
- **Interview Q&A:** "Add a new requirement live — walk me through it."

### Phase 7 — Deliverables (~4–5h)
- **Goal:** the three graded artifacts.
- **Build:** write-up PDF (Section 1 framing incl. why RAG not fine-tuning; Section 2 every choice + rejected alternative mapped to Q15–Q18 incl. CRAG/CAG; Section 3 results + error analysis; Section 4 one production consideration + one limitation); README reproduction pass in a clean venv; 5-min video script. Plus baked-in extras: freshness (incremental add), 1k-concurrency architecture (re-ranker CPU→GPU, LLM rate limits→cache/queue, FAISS flat→HNSW/pgvector), cost/latency table.
- **Concept brief:** communicating reasoning concisely; the write-up ↔ rubric mapping.

---

## Ablation methodology (OFAT — reference for Phase 3)

Baseline config, then vary one axis at a time:

| Axis | Variants | Answers |
|---|---|---|
| Chunking | recursive-256 / recursive-512 / structure-aware | Q15 |
| Embeddings | bge-small (local) / bge-base (local) / Bedrock embedding | Q16 |
| Retrieval | dense / BM25 / hybrid-RRF | Q17 |
| Re-ranking | none / cross-encoder | Q17 |

Report P@5, R@5, R@10, MRR, nDCG@10 per variant. Winner → Phase 5 e2e eval.

## Considered architectures (named alternatives for the write-up)

The 30% Algorithm-Selection dimension rewards naming, comparing, and rejecting alternatives. Our defensible set:

- **Dense-only / BM25-only vs hybrid-RRF** — decided by ablation (Q17); baseline retrievers named and beaten.
- **Corrective RAG (CRAG, Yan et al. 2024)** — adopt its retrieval-assess-and-correct *routing*; **reject its web-search fallback** (unsafe for legal — external content isn't the executed contract). Adapted as a within-corpus reformulate-or-abstain loop, measured (Phase 4/5).
- **Cache-Augmented Generation (CAG, Chan et al.)** — preload the whole contract, skip retrieval. Tempting (per-contract queries; kills retrieval miss). **Rejected as primary:** (1) off-track (Track D mandates a retrieval pipeline), (2) tail contracts exceed useful context + ~15× per-query token cost, (3) blurs verbatim-citation grounding. **Partially adopted:** prompt-cache the contract across a session (Bedrock prompt caching = cost/latency win); RAG-vs-CAG ablation on short contracts (Phase 5). Note: true KV-cache CAG needs a self-hosted model — via Bedrock it's prompt caching, not KV reuse.
- **Self-RAG** — reflection-token self-critique; needs fine-tuning → wrong fit for a no-fine-tune 48h build. Named, rejected.
- **Adaptive-RAG** — routes by query complexity; overkill when all queries are single-hop clause lookups. Named, rejected.
- **GraphRAG** — knowledge graph over entities; heavy, shines for multi-hop cross-document synthesis, not single-clause lookup in one contract. Named, rejected (future work: cross-contract obligations analysis).

## Write-up question mapping (Track D mandatory)

- **Q15 chunking** — decided by ablation; boundary-loss analysis + overlap/structure mitigation.
- **Q16 embeddings** — bge-small vs Bedrock embedding, compared **empirically** on P@k/R@k.
- **Q17 fusion/rerank lift** — dense vs BM25 vs hybrid-RRF vs +cross-encoder, one table, same eval set.
- **Q18 hallucination** — 4-layer defense: grounding prompt, deterministic verbatim verifier, score-threshold abstention, judge-sampled faithfulness (+ CRAG-adapted corrective loop).
- Plus: freshness, query handling, 1k-concurrency architecture, honest limitations (tables/exhibits, multi-span answers), and CRAG/CAG as considered alternatives.

## Verification (how we know each phase works)

- **Phase 0:** `.venv` activates; `pip install -r requirements.txt` succeeds; repo pushes to `Origin/main`.
- **Phase 1:** `build_eval_set.py` asserts every gold chunk contains its annotated span; prints label counts (answerable/unanswerable, per-category).
- **Phase 2:** round-trip test — chunk a known contract, confirm offsets reconstruct the source; index builds and returns neighbors.
- **Phase 3:** ablation tables reproduce from `configs/` with fixed seeds; sanity-check hybrid ≥ max(dense, BM25) on R@k or explain why not.
- **Phase 4:** verifier rejects a deliberately fabricated quote; pipeline abstains on a below-threshold query.
- **Phase 5:** e2e metrics + error buckets sum to the sample size (no dropped cases).
- **Phase 6:** `app.py` answers the 5 representative queries end-to-end (incl. 1 abstain, 1 OOS) with verifier-passing citations.
- **Phase 7:** README steps executed top-to-bottom in a fresh `.venv` before submission.

## Open items to confirm at build time (not blocking)

- Bedrock: confirm region + which Claude model IDs are access-enabled (Haiku-class preferred for cost); confirm Cohere/Titan embedding access for the Q16 comparison (fall back to bge-base local if not enabled).
- Git push auth to `github.com/gargisethi100/Avathon` (may prompt for credentials on first push).
