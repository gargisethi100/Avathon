"""
build_eval_set.py — construct the end-to-end (generation) evaluation sample.

Retrieval metrics run on ALL 1,244 answerable questions (gold = char spans in
qa_test.parquet), so they need no sampling. This script builds the *generation*
eval sample, which must be small to control LLM cost:

  - a seeded, category-stratified draw of ~150 answerable + ~150 unanswerable
    questions  ->  data/processed/e2e_sample.parquet
  - loads the hand-written out-of-scope + ambiguous queries (eval/oos_queries.json)

Design decision D-21: a *balanced* sample gives statistical power to BOTH
answer-F1 (answerable) and abstention precision/recall (unanswerable); the true
70%-unanswerable distribution is reported separately at eval time. See DECISIONS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "processed"
OOS = Path(__file__).resolve().parents[1] / "eval" / "oos_queries.json"
SEED = 42
N_PER_CLASS = 150


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Category-proportional draw of n rows (no groupby.apply -> no deprecation)."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = []
    for cat, count in df["category"].value_counts().items():
        sub = df[df["category"] == cat]
        k = min(count, max(1, round(count * frac)))
        parts.append(sub.sample(k, random_state=seed))
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    elif len(out) < n:
        extra = df.drop(out.index).sample(n - len(out), random_state=seed)
        out = pd.concat([out, extra])
    return out


def main() -> int:
    qa = pd.read_parquet(DATA / "qa_test.parquet")
    ans = qa[qa["is_answerable"]]
    una = qa[~qa["is_answerable"]]

    s_ans = stratified_sample(ans, N_PER_CLASS, SEED)
    s_una = stratified_sample(una, N_PER_CLASS, SEED)
    sample = (
        pd.concat([s_ans, s_una])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    sample.to_parquet(DATA / "e2e_sample.parquet", index=False)

    print(
        f"E2E sample: {len(sample)} = "
        f"{int(sample['is_answerable'].sum())} answerable + "
        f"{int((~sample['is_answerable']).sum())} unanswerable  (seed={SEED})"
    )
    print(f"  categories covered: {sample['category'].nunique()}/41")
    print(f"  contracts covered:  {sample['title'].nunique()}/102")

    oos = json.loads(OOS.read_text(encoding="utf-8"))
    print(
        f"OOS/ambiguous queries: {len(oos['out_of_scope'])} out-of-scope, "
        f"{len(oos['ambiguous'])} ambiguous"
    )
    print(f"\nWrote {DATA / 'e2e_sample.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
