"""
app.py — ClauseLens demo (Contract Intelligence).  Run:  streamlit run app.py

A polished, inspectable product UI over the unchanged pipeline. After a question:
  - LEFT  (≈62%): the answer + compact status badges + quote-verification status
  - RIGHT (≈38%): only the supporting evidence (cited chunks)
  - below: an optional "Retrieval diagnostics" panel with the full top-k trace
Conversational memory is preserved (follow-ups are contextualized). Rendering lives in
`src/ui_components.py`; this file is orchestration only.
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
EDGE_CASES = {"Out-of-scope query": "What is the capital of France?", "Ambiguous query": "What are the terms?"}

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
    for label, q in EDGE_CASES.items():
        if st.button(label, key=f"edge_{label}"):
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

# ---------------- header + ask ----------------
ui.render_header()
st.caption("recursive-512 chunks · hybrid dense + BM25 with RRF fusion · Claude Haiku 4.5 · quote-verified")
st.write("")

with st.form("ask", clear_on_submit=True):
    c1, c2 = st.columns([6, 1])
    typed = c1.text_input("Question", placeholder="Ask a question about this contract…",
                          label_visibility="collapsed")
    submitted = c2.form_submit_button("Ask", type="primary", use_container_width=True)

pending = st.session_state.pop("pending_q", None)
question = pending or (typed if submitted else None)

if question and question.strip():
    with st.spinner("Contextualize → gate → retrieve → generate → verify…"):
        result = session.ask(question.strip(), contract)
    st.session_state.turns.append((question.strip(), result))

# ---------------- render latest turn ----------------
turns = st.session_state.get("turns", [])
if not turns:
    st.info("Select a contract and ask a question — or try a suggestion in the sidebar.")
else:
    _, result = turns[-1]
    if result.get("was_rewritten"):
        st.caption(f"↻ interpreted as: *{result['standalone_question']}*")

    if result.get("status") == "answered":
        left, right = st.columns([62, 38], gap="large")
        with left:
            ui.render_answer(result)
        with right:
            ui.render_evidence(result)
    else:
        ui.render_state(result)

    ui.render_diagnostics(result)

    if len(turns) > 1:
        with st.expander(f"Conversation history ({len(turns) - 1} earlier)"):
            for qq, rr in reversed(turns[:-1]):
                st.markdown(f"**{qq}**  \n{rr.get('answer', '')[:180]}")
