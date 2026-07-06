# ClauseLens — RAG for Contract Intelligence

**Avathon AI Hiring Challenge · Scenario 2 (Gen AI for Enterprise Documents) · Track D (RAG / LLM Knowledge Systems)**

ClauseLens is a legal-team assistant that answers natural-language questions about a
contract corpus (CUAD) with **verbatim citations, faithfulness verification, and graceful
abstention**. Every major design choice (chunking, embeddings, retrieval mode, re-ranking)
is decided by a **measured ablation** against CUAD's expert span annotations — not by
assertion.

> **Status:** under active development, built phase by phase.
> See **[PLAN.md](PLAN.md)** for the full architecture, phased roadmap, and the reasoning
> behind every design choice.

## Setup (reproducible from scratch)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **Python:** 3.13.2
- **Generator LLM:** Claude on AWS Bedrock (set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, CPU)

## Reproduction

End-to-end reproduction steps are documented per phase as each lands.

## Data & License

Corpus: **CUAD** — Contract Understanding Atticus Dataset (Hendrycks et al., NeurIPS 2021),
CC BY 4.0. Sourced from Hugging Face: `theatticusproject/cuad-qa` (QA pairs) and
`theatticusproject/cuad` (raw contract text).
