"""
chunking.py — pluggable, offset-exact chunking for CUAD contracts.

Every chunk carries (char_start, char_end) into the *original* contract text, so
`contract_text[chunk.char_start:chunk.char_end] == chunk.text` exactly. This is the
linchpin that lets retrieval relevance be computed by overlapping a chunk's char
span with CUAD's gold answer span (Phase 1) — chunking-independent labels.

Strategies (the chunking axis of the Phase-3 ablation, Q15):
  - "recursive"        : fixed-size, boundary-respecting (separators hierarchy),
                         with overlap. Sizes given in *nominal tokens* (chars/4).
  - "structure_aware"  : split on numbered clause headings / ARTICLE / SECTION;
                         over-long sections fall back to recursive sub-splitting.
                         Falls back entirely to recursive if too few headings are
                         found (messy/unstructured contracts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

CHARS_PER_TOKEN = 4  # nominal; keeps chunking decoupled from any embedder tokenizer
_SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]

# Numbered clause headings, ARTICLE/Section markers at line start.
_HEADING_RE = re.compile(
    r"(?m)^[ \t]*("
    r"(?:\d+\.){1,}\d*"          # 1.  1.1  2.3.4
    r"|\d+\.(?=\s)"              # 1.  (single level)
    r"|ARTICLE\s+[\dIVXLC]+"     # ARTICLE IV
    r"|SECTION\s+\d+"            # SECTION 3
    r"|Section\s+\d+"            # Section 3
    r")\b"
)


@dataclass(frozen=True)
class Chunk:
    contract_id: str
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    token_estimate: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# core: turn text into offset spans
# --------------------------------------------------------------------------- #
def _atomize(text: str, target: int, separators: list[str], base: int = 0) -> list[tuple[int, int]]:
    """Recursively split into atoms each <= target chars, tracking char offsets."""
    if len(text) <= target:
        return [(base, base + len(text))] if text.strip() else []
    if not separators:  # hard split as last resort
        return [(base + i, base + min(i + target, len(text))) for i in range(0, len(text), target)]

    sep, rest = separators[0], separators[1:]
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        k = text.find(sep, start)
        if k == -1:
            piece, p_base, nxt = text[start:], base + start, len(text)
        else:
            piece, p_base, nxt = text[start:k], base + start, k + len(sep)
        if len(piece) > target:
            out.extend(_atomize(piece, target, rest, p_base))
        elif piece.strip():
            out.append((p_base, p_base + len(piece)))
        start = nxt
    return out


def _merge(atoms: list[tuple[int, int]], target: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack atoms into <=target-char spans with ~overlap-char overlap."""
    if not atoms:
        return []
    spans: list[tuple[int, int]] = []
    i, n = 0, len(atoms)
    while i < n:
        first = atoms[i][0]
        j = i
        while j < n and atoms[j][1] - first <= target:
            j += 1
        if j == i:
            j = i + 1
        end = atoms[j - 1][1]
        spans.append((first, end))
        if j >= n:
            break
        # start next chunk inside the last `overlap` chars of this one
        target_start = end - overlap
        ni = j
        for k in range(i, j):
            if atoms[k][0] >= target_start:
                ni = k
                break
        i = ni if ni > i else i + 1
    return spans


def _recursive_spans(text: str, target_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    return _merge(_atomize(text, target_chars, _SEPARATORS), target_chars, overlap_chars)


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _structure_spans(
    text: str, target_chars: int, overlap_chars: int, min_headings: int = 5
) -> list[tuple[int, int]]:
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if len(starts) < min_headings:  # too unstructured -> recursive fallback
        return _recursive_spans(text, target_chars, overlap_chars)

    bounds = starts + [len(text)]
    if starts[0] > 0:  # keep preamble before the first heading
        bounds = [0] + bounds

    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= target_chars:
            if text[a:b].strip():
                spans.append((a, b))
        else:  # over-long section -> sub-split, shifting offsets back to absolute
            for s, e in _recursive_spans(text[a:b], target_chars, overlap_chars):
                spans.append((a + s, a + e))
    return spans


def chunk_text(
    text: str,
    contract_id: str,
    strategy: str = "recursive",
    target_tokens: int = 256,
    overlap_frac: float = 0.15,
) -> list[Chunk]:
    """Chunk one contract. Returns offset-exact Chunk objects."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = int(target_chars * overlap_frac)

    if strategy == "recursive":
        spans = _recursive_spans(text, target_chars, overlap_chars)
    elif strategy == "structure_aware":
        spans = _structure_spans(text, target_chars, overlap_chars)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")

    chunks = []
    for i, (s, e) in enumerate(spans):
        piece = text[s:e]
        chunks.append(
            Chunk(
                contract_id=contract_id,
                chunk_id=f"{contract_id}::{strategy}::{i}",
                text=piece,
                char_start=s,
                char_end=e,
                token_estimate=max(1, round(len(piece) / CHARS_PER_TOKEN)),
            )
        )
    return chunks


STRATEGIES = ("recursive", "structure_aware")
