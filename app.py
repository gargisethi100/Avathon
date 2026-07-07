"""
app.py — ClauseLens conversational demo.  Run:  streamlit run app.py

A multi-turn chat over a selected contract. Each turn: the follow-up is rewritten
into a standalone question (condense-question memory), then run through the full
pipeline — query gate → hybrid-RRF retrieval → grounded Claude answer → deterministic
verifier — with the retrieved chunks, verified quotes, and a faithfulness badge shown.
Memory is bounded (last-5 turns + running summary) and resets on contract switch.
Winner config (pipeline DEFAULTS: recursive-512 + bge-small + hybrid-RRF), Claude Haiku 4.5.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from conversation import ClauseLensSession
from pipeline import ClauseLens

DATA = ROOT / "data" / "processed"
DEMO_CONTRACTS = 6

st.set_page_config(page_title="ClauseLens", layout="wide")


@st.cache_resource(show_spinner="Loading ClauseLens (embedding a few demo contracts)…")
def load_lens():
    contracts = pd.read_parquet(DATA / "contracts_test.parquet")
    demo = contracts.sample(DEMO_CONTRACTS, random_state=1).reset_index(drop=True)
    lens = ClauseLens(use_gate=True)  # winner config from pipeline DEFAULTS
    for _, c in demo.iterrows():
        lens.add_contract(c["title"], c["text"])
    return lens, list(demo["title"])


lens, titles = load_lens()

st.title("ClauseLens — conversational contract Q&A")
st.caption("recursive-512 + bge-small + hybrid-RRF · Claude Haiku 4.5 · memory via query contextualization")

with st.sidebar:
    st.header("Contract")
    contract = st.selectbox("Select a contract", titles, format_func=lambda t: t[:36] + "…")
    if st.button("🗑️ New conversation"):
        st.session_state.pop("session", None)
        st.session_state.pop("messages", None)
        st.rerun()
    st.markdown(
        "**Try a thread (memory in action):**\n"
        "1. What is the termination provision?\n"
        "2. *And what is its notice period?*\n"
        "3. Who are the parties?\n\n"
        "**Also:** *What is the capital of France?* (out-of-scope) · *What are the terms?* (ambiguous)"
    )

# Per-conversation state; reset on contract switch (contract-scoped memory).
if st.session_state.get("contract") != contract:
    st.session_state.contract = contract
    st.session_state.session = ClauseLensSession(lens, contract)
    st.session_state.messages = []

session: ClauseLensSession = st.session_state.session

# Render prior turns.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

if prompt := st.chat_input("Ask about the selected contract…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Contextualize → gate → retrieve → generate → verify…"):
            res = session.ask(prompt, contract)

        if res.get("was_rewritten"):
            st.caption(f"↻ interpreted as: *{res['standalone_question']}*")
        st.markdown(res["answer"])

        meta = []
        gate = res["trace"].get("gate")
        if gate:
            meta.append(f"gate={gate}")
        if res["status"] == "answered":
            verdict = res.get("verdict")
            meta.append("🟢 grounded" if verdict == "PASS" else f"🔴 {verdict}")
            if res.get("quotes") or res["trace"].get("retrieved"):
                with st.expander("Verified quotes + retrieved chunks (hybrid-RRF scores)"):
                    for q in res.get("quotes", []):
                        st.markdown(f"> {q}")
                    if res.get("quotes"):
                        st.divider()
                    chunks = {c.chunk_id: c for c in lens.index.stores[contract].chunks}
                    for cid_, score in res["trace"].get("retrieved", []):
                        ch = chunks.get(cid_)
                        st.markdown(f"`{score:.3f}`  {(ch.text[:220] + '…') if ch else '(missing)'}")
        meta_str = " · ".join(meta)
        if meta_str:
            st.caption(meta_str)

    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "meta": meta_str})
