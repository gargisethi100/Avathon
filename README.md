# ClauseLens — RAG for Contract Intelligence

**Avathon AI Hiring Challenge · Scenario 2 (Gen AI for Enterprise Documents) · Track D (RAG / LLM Knowledge Systems)**

**Problem.** Legal teams are drowning in unstructured commercial contracts. Answering a
routine question — *"what is the termination notice period in this agreement?"* — means
manually hunting through dozens of pages, and in a legal setting an unverifiable or
hallucinated AI answer is **worse than no answer at all**.

**Solution.** ClauseLens answers natural-language questions about a **specific contract** with
**verbatim citations, deterministic faithfulness verification, and graceful abstention**.
Every major design choice (chunking, embeddings, retrieval mode, re-ranking) is decided by a
**measured ablation** against CUAD's expert answer spans — not by assertion.

> **📹 Walkthrough video (5 min):** [loom.com/share/f0636110944647beb4830950167fc04a](https://www.loom.com/share/f0636110944647beb4830950167fc04a) · **📄 Write-up:** [ClauseLens — Technical Write-Up (Track D).pdf](<ClauseLens — Technical Write-Up (Track D).pdf>)

```
query ─▶ query gate (in-scope / out-of-scope / ambiguous)
      ─▶ hybrid retrieval (dense bge + BM25, RRF fusion, per-contract)
      ─▶ Claude Haiku 4.5 (grounded: quote verbatim · cite [n] · else NOT_FOUND)
      ─▶ deterministic verifier (every quote must be verbatim in context) + abstention
      ─▶ answer + citations + faithfulness verdict   |   graceful decline
```

## Key measured findings (see `results/`)

- **BM25 > dense** on this legal corpus (R@5 .54 vs .47) — queries are keyword-heavy; **hybrid-RRF wins** (.55).
- **A generic (MS-MARCO) cross-encoder rerank *hurt*** (.55→.48) — it mis-transfers to contract text. Shipped **without** it.
- **recursive-512 > recursive-256** (R@10 .75 vs .67); **structure-aware chunking did *not* beat recursive** (prior refuted).
- **bge-small ≈ bge-base** (within ~1 pt) — a free local model suffices.
- **Winner: recursive-512 + bge-small + hybrid-RRF.**

## Setup (reproducible from scratch)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **Python 3.13.2.** All deps pinned in `requirements.txt`.
- **Generator:** Claude on **AWS Bedrock**. Copy **[`.env.example`](.env.example)** to `.env`
  (gitignored) and fill in your Bedrock API key (or IAM key pair) + region.
  Enable **Anthropic Claude model access** in the Bedrock console for that region.
- **Corporate SSL note:** `truststore` routes TLS through the OS trust store (handles SSL-inspection proxies); a no-op elsewhere.

## Reproduce end-to-end

```bash
# 1. Data + gold labels (CUAD test split via HF Parquet; asserts every span matches its offset)
python data/download_cuad.py
python data/eda.py                 # corpus stats + figures -> results/
python data/build_eval_set.py      # balanced 150/150 e2e sample + OOS queries

# 2. Chunking integrity (offset round-trip + gold-span coverage = recall ceiling)
python eval/chunk_coverage.py       # -> results/chunk_coverage.md  (0 fails, 100% coverage)

# 3. Retrieval OFAT ablations (winner: recursive-512 + bge-small + hybrid-RRF)
python eval/retrieval_eval.py       # -> results/retrieval_metrics.{md,csv}

# 4. End-to-end eval (answer-F1, abstention P/R, faithfulness, error buckets) — needs Bedrock
python eval/e2e_eval.py             # -> results/e2e_metrics.md, e2e_rows.json

# 5. Interactive conversational demo (multi-turn; follow-ups resolved via query contextualization)
streamlit run app.py
```

> Retrieval ablations run on a seeded 20-contract subset for CPU tractability (full 102 ≈ 90 min);
> the relative ranking is what the ablation decides. Edit `SUBSET_N` in `eval/retrieval_eval.py` to scale.

## Repo structure

```
data/    download_cuad.py · build_eval_set.py · eda.py        # data + gold labels
src/     chunking · embeddings · indexing · retrieval          # ingestion + retrieval
         generation · verifier · query_gate · pipeline · config # generation + defense
eval/    chunk_coverage · retrieval_eval · e2e_eval            # measurement
results/ metrics tables, figures, traces
app.py   Streamlit demo
ClauseLens — Technical Write-Up (Track D).pdf   # 1–2 page write-up
```

## Data & License

**CUAD** — Contract Understanding Atticus Dataset (Hendrycks et al., NeurIPS 2021), CC BY 4.0.
Sourced from Hugging Face `theatticusproject/cuad-qa` (SQuAD-2.0-style QA with expert spans).
Downloaded data, indexes, and `.env` are gitignored (regenerable / secret).
