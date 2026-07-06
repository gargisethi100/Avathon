"""
download_cuad.py — fetch CUAD-QA (test split) and normalize it to disk.

Why Parquet and not `load_dataset("theatticusproject/cuad-qa")`?
  CUAD-QA ships as a *dataset loading script* (`cuad-qa.py`). `datasets>=5` refuses
  to run dataset scripts, so we pull HF's auto-converted Parquet
  (revision `refs/convert/parquet`) directly — reproducible and script-free.

Corporate-network note:
  This machine sits behind an SSL-inspection proxy whose root CA is only in the
  Windows trust store, so plain `certifi` fails with CERTIFICATE_VERIFY_FAILED.
  We route TLS through the OS trust store via `truststore` (no-op elsewhere).

Outputs (data/processed/, gitignored — regenerable from this script):
  contracts_test.parquet : one row per contract  [title, text, n_chars]
  qa_test.parquet        : one row per question
                           [id, title, category, question, answer_texts,
                            answer_starts, answer_ends, is_answerable]

The script also validates the linchpin invariant: every annotated answer span must
match the contract text exactly at its recorded char offset. If it doesn't, the
gold retrieval labels would be corrupt, so we fail loudly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Route TLS through the OS trust store (corporate SSL-inspection CA lives there).
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - non-corporate networks don't need it
    pass

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "theatticusproject/cuad-qa"
REVISION = "refs/convert/parquet"
TEST_PARQUET = "default/test/0000.parquet"
OUT = Path(__file__).resolve().parent / "processed"


def parse_category(qid: str) -> str:
    """CUAD ids are '{title}__{category}'."""
    return qid.rsplit("__", 1)[-1] if "__" in qid else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO}:{TEST_PARQUET} @ {REVISION} ...")
    path = hf_hub_download(
        REPO, TEST_PARQUET, repo_type="dataset", revision=REVISION
    )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} QA rows across {df['title'].nunique()} contracts.")

    # --- normalize QA rows + validate spans ---
    recs = []
    span_checks = span_mismatches = 0
    mismatch_examples = []
    for _, r in df.iterrows():
        ctx = r["context"]
        ans = r["answers"]
        texts = [str(t) for t in ans["text"]]
        starts = [int(s) for s in ans["answer_start"]]
        ends = []
        for t, s in zip(texts, starts):
            e = s + len(t)
            ends.append(e)
            span_checks += 1
            if ctx[s:e] != t:
                span_mismatches += 1
                if len(mismatch_examples) < 3:
                    mismatch_examples.append((r["id"], t, ctx[s:e]))
        recs.append(
            {
                "id": r["id"],
                "title": r["title"],
                "category": parse_category(r["id"]),
                "question": r["question"],
                "answer_texts": texts,
                "answer_starts": starts,
                "answer_ends": ends,
                "is_answerable": len(texts) > 0,
            }
        )
    qa = pd.DataFrame(recs)

    # --- unique contracts ---
    contracts = (
        df.groupby("title")["context"].first().reset_index().rename(columns={"context": "text"})
    )
    contracts["n_chars"] = contracts["text"].str.len()

    # --- linchpin: gold-span integrity ---
    ok = span_checks - span_mismatches
    print(f"\nGold-span integrity: {ok}/{span_checks} spans match context exactly.")
    if span_mismatches:
        print("  mismatch examples (id, annotated, context_at_offset):")
        for ex in mismatch_examples:
            print("   ", ex)
    assert span_mismatches == 0, (
        f"{span_mismatches} answer spans do not match their char offsets; "
        "gold labels would be corrupt — investigate before proceeding."
    )

    # --- stats ---
    n_ans = int(qa["is_answerable"].sum())
    cl = contracts["n_chars"]
    print(
        f"Contracts: {len(contracts)} | QA: {len(qa)} | "
        f"answerable: {n_ans} | unanswerable: {len(qa) - n_ans} "
        f"({100 * (len(qa) - n_ans) / len(qa):.1f}% unanswerable)"
    )
    print(f"Clause categories: {qa['category'].nunique()}")
    print(
        f"Contract length (chars): min={int(cl.min())} "
        f"median={int(cl.median())} max={int(cl.max())}"
    )

    # --- save ---
    contracts.to_parquet(OUT / "contracts_test.parquet", index=False)
    qa.to_parquet(OUT / "qa_test.parquet", index=False)
    print(f"\nWrote:\n  {OUT / 'contracts_test.parquet'}\n  {OUT / 'qa_test.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
