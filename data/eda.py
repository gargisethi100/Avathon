"""
eda.py — corpus EDA for the write-up.

Deterministic and file-based (a script, not a notebook: reproducible, diff-able,
no hidden execution-order state). Reads the normalized Parquet from
download_cuad.py and writes figures + a stats summary to results/.

Token counts are estimated as chars/4 (rough English heuristic) and labeled as
estimates — enough to reason about context-window fit for the CAG analysis (D-13).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(__file__).resolve().parent / "processed"
RESULTS = Path(__file__).resolve().parents[1] / "results"
CHARS_PER_TOKEN = 4


def main() -> int:
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    qa = pd.read_parquet(DATA / "qa_test.parquet")
    RESULTS.mkdir(parents=True, exist_ok=True)

    contracts["est_tokens"] = (contracts["n_chars"] / CHARS_PER_TOKEN).round().astype(int)
    qa["n_spans"] = qa["answer_texts"].apply(len)

    lines: list[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    n_ans = int(qa["is_answerable"].sum())
    emit("# CUAD-QA (test split) — EDA summary\n")
    emit(f"- Contracts: {len(contracts)}")
    emit(
        f"- QA pairs: {len(qa)} "
        f"(answerable {n_ans}, unanswerable {len(qa) - n_ans}, "
        f"{100 * (len(qa) - n_ans) / len(qa):.1f}% unanswerable)"
    )
    emit(f"- Clause categories: {qa['category'].nunique()}")
    cl, tk = contracts["n_chars"], contracts["est_tokens"]
    emit(
        f"- Contract chars: min {int(cl.min())}, median {int(cl.median())}, "
        f"mean {int(cl.mean())}, max {int(cl.max())}"
    )
    emit(f"- Est. tokens (chars/{CHARS_PER_TOKEN}): median {int(tk.median())}, max {int(tk.max())}")
    for thr in (8000, 32000, 128000):
        emit(f"  - contracts <= {thr} tokens: {100 * (tk <= thr).mean():.1f}% (CAG context-fit)")
    sp = qa.loc[qa.is_answerable, "n_spans"]
    emit(f"- Spans per answerable question: mean {sp.mean():.2f}, max {int(sp.max())}")

    cat = qa.groupby("category")["is_answerable"].mean().sort_values()
    emit("\n## Rarest clauses (lowest presence)")
    for c, v in cat.head(5).items():
        emit(f"  - {c}: {100 * v:.1f}% of contracts")
    emit("## Most common clauses")
    for c, v in cat.tail(5).items():
        emit(f"  - {c}: {100 * v:.1f}% of contracts")

    (RESULTS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Figure 1 — contract length (tokens)
    plt.figure(figsize=(7, 4))
    plt.hist(tk, bins=40, color="#4C78A8")
    plt.axvline(8000, color="#F58518", ls="--", label="8k tokens")
    plt.axvline(128000, color="#E45756", ls="--", label="128k tokens")
    plt.xlabel("estimated tokens per contract")
    plt.ylabel("# contracts")
    plt.title("CUAD contract length (test split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "eda_contract_length.png", dpi=120)
    plt.close()

    # Figure 2 — clause prevalence
    plt.figure(figsize=(7, 9))
    cat.plot(kind="barh", color="#4C78A8")
    plt.xlabel("fraction of contracts where the clause is present")
    plt.title("Clause category prevalence (answerable rate)")
    plt.tight_layout()
    plt.savefig(RESULTS / "eda_category_prevalence.png", dpi=120)
    plt.close()

    print(f"\nWrote figures + summary to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
