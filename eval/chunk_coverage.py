"""
chunk_coverage.py — chunking diagnostics, independent of any embedder.

Reports, per chunking config:
  - roundtrip_fail : chunks where text[start:end] != chunk.text (must be 0)
  - gold_coverage  : fraction of gold answer spans overlapping >=1 chunk
                     (== the retrieval RECALL CEILING for that chunking)
  - chunk count and token size stats

If gold_coverage < 100%, some answers are unreachable no matter how good the
retriever is — so this bounds what Phase-3 recall can possibly achieve.
Writes results/chunk_coverage.md.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from chunking import STRATEGIES, chunk_text

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def main() -> int:
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    qa = pd.read_parquet(DATA / "qa_test.parquet")

    gold: dict[str, list[tuple[int, int]]] = {}
    for _, r in qa[qa["is_answerable"]].iterrows():
        gold.setdefault(r["title"], []).extend(zip(r["answer_starts"], r["answer_ends"]))

    lines = [
        "# Chunking diagnostics (CUAD test split)\n",
        "| strategy | tokens | chunks | roundtrip_fail | gold_coverage | mean_tok | max_tok |",
        "|---|---|---|---|---|---|---|",
    ]
    for strat in STRATEGIES:
        for tt in (256, 512):
            total = rt = ghit = gtot = 0
            sizes: list[int] = []
            for _, c in contracts.iterrows():
                text, cid = c["text"], c["title"]
                chunks = chunk_text(text, cid, strategy=strat, target_tokens=tt)
                total += len(chunks)
                sizes.extend(ch.token_estimate for ch in chunks)
                rt += sum(text[ch.char_start : ch.char_end] != ch.text for ch in chunks)
                for gs, ge in gold.get(cid, []):
                    gtot += 1
                    if any(overlaps(gs, ge, ch.char_start, ch.char_end) for ch in chunks):
                        ghit += 1
            lines.append(
                f"| {strat} | {tt} | {total} | {rt} | {100 * ghit / gtot:.2f}% | "
                f"{statistics.mean(sizes):.0f} | {max(sizes)} |"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "chunk_coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS / 'chunk_coverage.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
