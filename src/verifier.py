"""
verifier.py — hallucination defense layers 2 & 3 (D-11, write-up Q18).

Layer 2 (deterministic, free, hard gate): every quoted span in the answer must be a
verbatim substring of the retrieved context (whitespace/case-normalized). A quote
that isn't grounded => the answer is FLAGGED/refused. This catches fabricated
support with zero LLM cost and no false negatives from a judge.

Layer 3 (score-threshold abstention): if the best retrieval score is below a
threshold, prefer abstention over a guess (threshold tuned in Phase 5).

Verdicts:
  PASS       — abstained, OR answered with >=1 quote and all quotes grounded
  UNGROUNDED — answered but a quote is NOT in the retrieved context (hallucination)
  NO_QUOTE   — answered but cited no verbatim quote (grounding not demonstrable)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# LLMs silently tidy punctuation when quoting (e.g. source "China ,otherwise" ->
# quote "China, otherwise"). Normalizing whitespace AND spaces around punctuation
# keeps the gate strict against fabrication while not false-flagging benign
# reformatting — critical so the Phase-5 faithfulness metric isn't under-reported.
_PUNCT_RE = re.compile(r"\s*([" + re.escape(r",.;:!?()[]\"'/-") + r"])\s*")


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.lower())
    s = _PUNCT_RE.sub(r"\1", s)
    return s.strip()


def _context_text(chunks) -> str:
    parts = [(c[0] if isinstance(c, tuple) else c).text for c in chunks]
    return _norm(" ".join(parts))


@dataclass
class Verification:
    verdict: str
    all_grounded: bool
    per_quote: list  # [(quote, grounded_bool)]

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def verify_answer(answer, chunks) -> Verification:
    ctx = _context_text(chunks)
    per_quote = [(q, _norm(q) in ctx) for q in answer.quotes]

    if answer.abstained:
        return Verification("PASS", True, per_quote)
    if not per_quote:
        return Verification("NO_QUOTE", False, per_quote)
    all_grounded = all(g for _, g in per_quote)
    return Verification("PASS" if all_grounded else "UNGROUNDED", all_grounded, per_quote)


def should_abstain(hits, threshold: float) -> bool:
    """True if the top retrieval score is below `threshold` (score-gated abstention)."""
    if not hits:
        return True
    top = hits[0][1] if isinstance(hits[0], tuple) else None
    return top is not None and top < threshold
