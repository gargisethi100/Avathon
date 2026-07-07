"""
app.py — ClauseLens demo (Contract Intelligence), conversational chatbot.

A persistent multi-turn chat thread: questions and grounded answers STACK like a
chatbot, follow-ups are resolved via conversational memory, and each assistant turn
is a rich, inspectable block — status badges, source-chip citations, supporting
evidence, quote-verification, and a collapsed retrieval-diagnostics panel. Rendering
lives in `src/ui_components.py`; this file is orchestration only. Backend unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import ui_components as ui
from conversation import ClauseLensSession
from pipeline import ClauseLens

DATA = ROOT / "data" / "processed"
DEMO_CONTRACTS = 6

SUGGESTIONS = ["What is the governing law?", "What is the termination notice period?", "Who are the parties?"]
EDGE_CASES = [
    ("What is the capital of France?", "Tests out-of-scope handling — the gate should decline it"),
    ("What are the terms?", "Tests ambiguous-query handling — the gate should ask to clarify"),
]

st.set_page_config(page_title="ClauseLens · Contract Intelligence", layout="wide")
ui.inject_css()


@st.cache_resource(show_spinner="Loading ClauseLens (embedding demo contracts)…")
def load_lens():
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    demo = contracts.sample(DEMO_CONTRACTS, random_state=1).reset_index(drop=True)
    lens = ClauseLens(use_gate=True)  # Phase-3 winner config from pipeline DEFAULTS
    for _, c in demo.iterrows():
        lens.add_contract(c["title"], c["text"])
    return lens, list(demo["title"])


lens, titles = load_lens()

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### Contract")
    contract = st.selectbox("Contract", titles, format_func=ui.short_name,
                            label_visibility="collapsed", help="Full CUAD document identifier")
    st.caption(contract[:58] + ("…" if len(contract) > 58 else ""))

    st.markdown("**Suggested questions**")
    for i, s in enumerate(SUGGESTIONS):
        if st.button(s, key=f"sug{i}"):
            st.session_state.pending_q = s

    st.markdown("**Edge-case tests**")
    for i, (q, tip) in enumerate(EDGE_CASES):
        if st.button(q, key=f"edge{i}", help=tip):
            st.session_state.pending_q = q

    st.divider()
    if st.button("New conversation", key="reset"):
        for k in ("session", "turns", "contract_key"):
            st.session_state.pop(k, None)
        st.rerun()

# ---------------- session (memory); reset on contract switch ----------------
if st.session_state.get("contract_key") != contract:
    st.session_state.contract_key = contract
    st.session_state.session = ClauseLensSession(lens, contract)
    st.session_state.turns = []  # list[(question, result)]
session: ClauseLensSession = st.session_state.session

# ---------------- header ----------------
ui.render_header()
st.caption("recursive-512 chunks · hybrid dense + BM25 with RRF fusion · Claude Haiku 4.5 · quote-verified")

# ---------------- chat thread (prior turns stack) ----------------
for q, result in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        ui.render_assistant_turn(result)

# ---------------- input (pinned bottom) + chips ----------------
prompt = st.chat_input(f"Ask about {ui.short_name(contract)}…")
question = st.session_state.pop("pending_q", None) or prompt

if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Contextualize → gate → retrieve → generate → verify…"):
            result = session.ask(question, contract)
        ui.render_assistant_turn(result)
    st.session_state.turns.append((question, result))
elif not st.session_state.turns:
    st.info("Ask a question about the selected contract — or try a suggestion in the sidebar. "
            "Follow-ups work: try “What is the termination provision?” then “and its notice period?”")
