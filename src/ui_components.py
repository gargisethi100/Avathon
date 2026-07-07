"""
ui_components.py — presentation layer for the ClauseLens demo.

Pure rendering helpers (no retrieval/generation logic): CSS, status badges, the
answer card, evidence cards, the diagnostics panel, and query-state messages. `app.py`
orchestrates; this module renders a pipeline `result` dict. Keeping them separate is
the whole point — the UI shows only supporting evidence by default and exposes the
full top-k retrieval trace behind a diagnostics expander.
"""
from __future__ import annotations

import html
import re

import streamlit as st

ACCENT = "#4F46E5"

_CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
  section[data-testid="stSidebar"] { width: 292px !important; min-width: 292px !important; }
  section[data-testid="stSidebar"] .stButton > button {
      padding: .28rem .6rem; font-size: .82rem; border-radius: 8px; width: 100%;
      border: 1px solid #E2E8F0; background: #fff; color: #1E293B; font-weight: 500; }
  section[data-testid="stSidebar"] .stButton > button:hover { border-color: #4F46E5; color: #4F46E5; }
  /* bordered containers -> clean cards */
  div[data-testid="stVerticalBlockBorderWrapper"] {
      background: #fff; border: 1px solid #E7E9EE !important; border-radius: 11px;
      box-shadow: 0 1px 2px rgba(16,24,40,.04); }
  .cl-h1 { font-size: 1.5rem; font-weight: 750; margin: 0; color: #0F172A; letter-spacing: -.01em; }
  .cl-sub { color: #64748B; margin: .15rem 0 0; font-size: .92rem; }
  .cl-section { font-size: .74rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
      color: #94A3B8; margin: .1rem 0 .55rem; }
  .cl-badges { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 .7rem; }
  .cl-badge { font-size: .72rem; font-weight: 600; padding: 2px 9px; border-radius: 6px; border: 1px solid transparent; }
  .cl-ok   { background: #ECFDF3; color: #067647; border-color: #ABEFC6; }
  .cl-info { background: #EEF2FF; color: #3730A3; border-color: #C7D2FE; }
  .cl-warn { background: #FFFAEB; color: #B54708; border-color: #FEDF89; }
  .cl-bad  { background: #FEF3F2; color: #B42318; border-color: #FECDCA; }
  .cl-muted{ background: #F2F4F7; color: #475467; border-color: #E4E7EC; }
  .cl-cite { font-size: .72rem; font-weight: 600; color: #3730A3; background: #EEF2FF;
      border: 1px solid #C7D2FE; border-radius: 5px; padding: 0 6px; margin: 0 2px; white-space: nowrap; }
  .cl-src  { font-weight: 700; color: #0F172A; font-size: .92rem; }
  .cl-meta { color: #64748B; font-size: .78rem; margin-top: 1px; }
  .cl-quote { border-left: 3px solid #4F46E5; background: #F7F8FE; padding: .5rem .7rem;
      border-radius: 0 7px 7px 0; margin: .4rem 0; color: #1E293B; font-size: .9rem; line-height: 1.45; }
  .cl-verify { border-radius: 8px; padding: .5rem .7rem; font-size: .84rem; margin-top: .7rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def short_name(title: str) -> str:
    """Readable display name from a CUAD filename-style title."""
    parts = [p for p in title.replace("-", "_").split("_") if p]
    if not parts:
        return title
    company = parts[0]
    tail = next((p for p in reversed(parts) if any(c.isalpha() for c in p) and not p.isdigit()), parts[-1])
    return f"{company[:22]} · {tail[:26]}" if tail != company else company[:40]


def render_header() -> None:
    st.markdown(
        '<div class="cl-h1">Contract Intelligence</div>'
        '<div class="cl-sub">Grounded question answering over legal agreements</div>',
        unsafe_allow_html=True,
    )


def render_badges(items) -> None:
    """items: list of (label, kind) where kind in ok/info/warn/bad/muted."""
    if not items:
        return
    pills = "".join(f'<span class="cl-badge cl-{k}">{html.escape(l)}</span>' for l, k in items)
    st.markdown(f'<div class="cl-badges">{pills}</div>', unsafe_allow_html=True)


def _cite_html(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r'<span class="cl-cite">Source \1</span>', text)


def _answer_badges(result: dict):
    badges = []
    gate = result.get("trace", {}).get("gate")
    if gate == "IN_SCOPE":
        badges.append(("In scope", "info"))
    n = len(result.get("evidence", []))
    if n:
        badges.append((f"{n} source{'s' if n != 1 else ''}", "info"))
    verdict = result.get("verdict")
    if result.get("abstained"):
        badges.append(("Abstained", "warn"))
    elif verdict == "PASS":
        badges.append(("Quotes verified", "ok"))
    elif verdict == "UNGROUNDED":
        badges.append(("Unverified quote", "bad"))
    elif verdict == "NO_QUOTE":
        badges.append(("No verbatim quote", "warn"))
    return badges


def _render_verification(result: dict) -> None:
    verdict = result.get("verdict")
    if result.get("abstained"):
        return
    conf = {
        "PASS": ("cl-ok", "✓ Verified — every cited quote matched the retrieved evidence verbatim."),
        "UNGROUNDED": ("cl-bad", "✕ Verification failed — a cited quote was not found verbatim in the retrieved evidence."),
        "NO_QUOTE": ("cl-warn", "! No verbatim quote — the answer did not cite an exact excerpt, so grounding can't be confirmed."),
    }.get(verdict)
    if conf:
        cls, msg = conf
        st.markdown(f'<div class="cl-verify {cls}">{msg}</div>', unsafe_allow_html=True)


def render_answer(result: dict) -> None:
    with st.container(border=True):
        st.markdown('<div class="cl-section">Answer</div>', unsafe_allow_html=True)
        render_badges(_answer_badges(result))
        st.markdown(_cite_html(result.get("answer", "")), unsafe_allow_html=True)
        _render_verification(result)


def render_state(result: dict) -> None:
    """Non-answered states: out-of-scope / ambiguous / abstained."""
    conf = {
        "declined_out_of_scope": ("Out of scope", "bad",
            "This question is outside the selected contract. Ask about a clause, party, date, or obligation in this document."),
        "needs_clarification": ("Ambiguous", "warn", result.get("answer", "")),
        "abstained_low_score": ("Abstained", "warn",
            "No sufficiently relevant clause was retrieved, so the system declined to answer rather than guess."),
    }.get(result.get("status"))
    with st.container(border=True):
        st.markdown('<div class="cl-section">Answer</div>', unsafe_allow_html=True)
        if conf:
            label, kind, msg = conf
            render_badges([(label, kind)])
            st.markdown(msg)
        else:
            st.markdown(result.get("answer", ""))


def render_evidence(result: dict) -> None:
    st.markdown('<div class="cl-section">Supporting evidence</div>', unsafe_allow_html=True)
    evidence = result.get("evidence", [])
    if not evidence:
        st.markdown('<div class="cl-meta">No specific supporting excerpt for this answer.</div>',
                    unsafe_allow_html=True)
        return
    for e in evidence:
        with st.container(border=True):
            meta = f"Chunk {e['chunk_index']}" if e.get("chunk_index") is not None else "Chunk"
            if e.get("char_start") is not None:
                meta += f" · chars {e['char_start']:,}–{e['char_end']:,}"
            st.markdown(f'<div class="cl-src">Source {e["source"]}</div>'
                        f'<div class="cl-meta">{html.escape(meta)}</div>', unsafe_allow_html=True)
            shown = e.get("quotes") or [e.get("excerpt", "")[:280].rstrip() + "…"]
            for q in shown:
                st.markdown(f'<div class="cl-quote">{html.escape(q)}</div>', unsafe_allow_html=True)
            score = e.get("reranker_score")
            label = "Reranker score" if score is not None else "RRF score"
            if score is None:
                score = e.get("rrf_score")
            if score is not None:
                st.markdown(f'<div class="cl-meta">{label}: {score:.3f} · final rank {e.get("final_rank")}</div>',
                            unsafe_allow_html=True)
            with st.expander("Full chunk"):
                st.markdown(e.get("excerpt", ""))


def render_diagnostics(result: dict) -> None:
    tr = result.get("trace", {})
    records = tr.get("retrieval_detail")
    with st.expander("🔬 Retrieval diagnostics — full top-k trace", expanded=False):
        st.markdown(f"**Pipeline:** {tr.get('pipeline', '—')}")
        if tr.get("gate"):
            st.caption(f"query gate = {tr['gate']}   ·   contract = {tr.get('contract_id', '')}")
        lat = tr.get("latency", {})
        if lat:
            st.caption("latency (ms) — " + " · ".join(f"{k.replace('_ms', '')}: {v}" for k, v in lat.items()))
        if records:
            import pandas as pd

            df = pd.DataFrame([{
                "Chunk": r["chunk_index"],
                "Dense": r["dense_rank"], "BM25": r["bm25_rank"], "RRF": r["rrf_rank"],
                "Rerank": (round(r["reranker_score"], 3) if r["reranker_score"] is not None else None),
                "Final": r["final_rank"],
                "Selected": "✓" if r["selected"] else "",
                "Preview": (r["text"][:90] + "…"),
            } for r in records])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No retrieval performed for this query (gated before retrieval).")


def render_assistant_turn(result: dict) -> None:
    """Render one assistant turn INSIDE a chat message (the bubble is the card).

    Composes the rich, inspectable answer block: rewrite note, status badges, answer
    with source-chip citations, quote-verification line, supporting evidence (default
    visible), and a collapsed diagnostics panel — per chat turn.
    """
    if result.get("was_rewritten"):
        st.caption(f"↻ interpreted as: {result.get('standalone_question', '')}")

    status = result.get("status")
    if status == "answered":
        render_badges(_answer_badges(result))
        st.markdown(_cite_html(result.get("answer", "")), unsafe_allow_html=True)
        _render_verification(result)
        if result.get("evidence"):
            render_evidence(result)
    else:
        conf = {
            "declined_out_of_scope": ("Out of scope", "bad",
                "This question is outside the selected contract. Ask about a clause, party, date, or obligation in this document."),
            "needs_clarification": ("Ambiguous", "warn", result.get("answer", "")),
            "abstained_low_score": ("Abstained", "warn",
                "No sufficiently relevant clause was retrieved, so the system declined to answer rather than guess."),
        }.get(status, (None, None, result.get("answer", "")))
        label, kind, msg = conf
        if label:
            render_badges([(label, kind)])
        st.markdown(msg)

    if result.get("trace", {}).get("retrieval_detail"):
        render_diagnostics(result)
