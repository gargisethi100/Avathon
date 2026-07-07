"""
e2e_eval.py — end-to-end generation eval on the balanced 300-pair sample.

Runs the ClauseLens pipeline per QA and reports (write-up Section 3):
  - answerable: token-F1 of the answer's supporting quotes vs the gold span text
    (extractive), plus quote-overlaps-gold rate
  - unanswerable: abstention precision / recall
  - faithfulness: deterministic verifier PASS rate over answered items
  - error buckets: retrieval_miss / correct_context_wrong_answer / unfaithful_quote /
    wrongful_abstention / wrongful_answer

The CUAD query gate adds no signal here (every CUAD question is a contract
question -> IN_SCOPE), so we run with use_gate=False to halve Bedrock calls; the
gate is tested separately on eval/oos_queries.json. Config-driven: plug in the
Phase-3 winning (chunking, embedder, mode). Writes results/e2e_metrics.md.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from indexing import tokenize
from pipeline import ClauseLens

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
_WORD = re.compile(r"[a-z0-9]+")


def token_f1(pred: str, gold: str) -> float:
    p, g = _WORD.findall(pred.lower()), _WORD.findall(gold.lower())
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstained(res: dict) -> bool:
    return res.get("abstained", False) or res["status"] in (
        "declined_out_of_scope", "needs_clarification", "abstained_low_score",
    )


def run(config: dict, sample_n: int | None = None, max_contracts: int | None = None) -> dict:
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    sample = pd.read_parquet(DATA / "e2e_sample.parquet")
    if sample_n:
        sample = sample.groupby("is_answerable", group_keys=False).head(sample_n // 2)

    needed = set(sample["title"])
    if max_contracts:
        needed = set(list(needed)[:max_contracts])
        sample = sample[sample["title"].isin(needed)].reset_index(drop=True)

    lens = ClauseLens(use_gate=False, **config)
    ctext = contracts.set_index("title")["text"].to_dict()
    for cid in needed:
        lens.add_contract(cid, ctext[cid])
    print(f"indexed {len(needed)} contracts, {lens.index.n_chunks} chunks", flush=True)

    rows = []
    for j, (_, r) in enumerate(sample.iterrows()):
        if j % 25 == 0:
            print(f"  [{j}/{len(sample)}]", flush=True)
        res = lens.answer(r["question"], r["title"], k=5)
        abstained = is_abstained(res)
        rec = {"id": r["id"], "question": r["question"], "answerable": bool(r["is_answerable"]),
               "abstained": abstained, "verdict": res.get("verdict"),
               "answer": (res.get("answer") or "")[:400],
               "retrieved": [cid for cid, _ in res["trace"].get("retrieved", [])]}
        if r["is_answerable"]:
            gold = " ".join(r["answer_texts"])
            pred = " ".join(res.get("quotes") or []) or res.get("answer", "")
            rec["gold"] = gold[:300]
            rec["f1"] = token_f1(pred, gold)
            gold_toks = set(_WORD.findall(gold.lower()))
            pred_toks = set(_WORD.findall(pred.lower()))
            rec["quote_hits_gold"] = len(gold_toks & pred_toks) / max(1, len(gold_toks)) >= 0.5
        rows.append(rec)
    return {"config": config, "rows": rows}


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]

    # abstention P/R (positive class = "should abstain" = unanswerable)
    tp = sum(1 for r in una if r["abstained"])
    fp = sum(1 for r in ans if r["abstained"])
    fn = sum(1 for r in una if not r["abstained"])
    ab_p = tp / (tp + fp) if (tp + fp) else 0.0
    ab_r = tp / (tp + fn) if (tp + fn) else 0.0

    faithful = sum(1 for r in answered if r["verdict"] == "PASS")
    answered_ans = [r for r in ans if not r["abstained"]]
    return {
        "n": len(rows), "answerable": len(ans), "unanswerable": len(una),
        "answer_f1": round(sum(r.get("f1", 0.0) for r in ans) / max(1, len(ans)), 4),
        "answer_f1_when_answered": round(sum(r.get("f1", 0.0) for r in answered_ans) / max(1, len(answered_ans)), 4),
        "quote_hits_gold": round(sum(1 for r in ans if r.get("quote_hits_gold")) / max(1, len(ans)), 4),
        "abstention_precision": round(ab_p, 4),
        "abstention_recall": round(ab_r, 4),
        "faithfulness_pass_rate": round(faithful / max(1, len(answered)), 4),
    }


def error_buckets(rows: list[dict]) -> dict:
    b = Counter()
    for r in rows:
        if r["answerable"]:
            if r["abstained"]:
                b["wrongful_abstention"] += 1
            elif r["verdict"] != "PASS":
                b["unfaithful_quote"] += 1
            elif r.get("f1", 0) < 0.3:
                b["correct_context_wrong_answer"] += 1
        else:  # unanswerable
            if not r["abstained"]:
                b["wrongful_answer"] += 1
    return dict(b)


if __name__ == "__main__":
    # Phase-3 measured winner.
    cfg = dict(chunking="recursive", tokens=512, embedder="bge-small", mode="hybrid")
    out = run(cfg, max_contracts=None)
    s = summarize(out["rows"])
    buckets = error_buckets(out["rows"])
    RESULTS.mkdir(exist_ok=True)
    lines = [f"# End-to-end eval ({s['n']} QA: {s['answerable']} answerable / {s['unanswerable']} unanswerable)\n",
             f"- config: {cfg}",
             f"- answer token-F1 (all answerable): {s['answer_f1']}",
             f"- answer token-F1 (when it answers): {s['answer_f1_when_answered']}",
             f"- quote-overlaps-gold rate: {s['quote_hits_gold']}",
             f"- abstention precision / recall: {s['abstention_precision']} / {s['abstention_recall']}",
             f"- faithfulness (verifier PASS rate): {s['faithfulness_pass_rate']}",
             f"- error buckets: {buckets}"]
    (RESULTS / "e2e_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "e2e_rows.json").write_text(json.dumps(out["rows"], indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
