"""
app.py — ClauseLens Streamlit demo.  Run:  streamlit run app.py

Shows the full pipeline transparently: query gate -> hybrid-RRF retrieval ->
grounded Claude answer -> deterministic verifier, with the retrieved chunks +
scores, verified quotes, a faithfulness badge, and graceful abstention / OOS
handling. Uses the Phase-3 winning config (pipeline DEFAULTS: recursive-512 +
bge-small + hybrid-RRF) and Claude Haiku 4.5 on Bedrock.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
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

st.title("ClauseLens — contract Q&A with citations & abstention")
st.caption("Track D · recursive-512 + bge-small + hybrid-RRF (rerank measured to hurt) · Claude Haiku 4.5 on Bedrock")

with st.sidebar:
    st.header("Contract")
    contract = st.selectbox("Select a contract", titles, format_func=lambda t: t[:38] + "…")
    st.markdown("**Try these:**")
    st.markdown(
        "- What is the governing law?\n"
        "- What is the termination notice period?\n"
        "- Who are the parties?\n"
        "- _What is the capital of France?_ (out-of-scope)\n"
        "- _What are the terms?_ (ambiguous)"
    )

question = st.text_input("Ask a question about the selected contract:")

if question:
    with st.spinner("Gate → retrieve → generate → verify…"):
        res = lens.answer(question, contract, k=5)
    status = res["status"]

    badge = {
        "answered": "✅ Answered",
        "declined_out_of_scope": "🚫 Out of scope — declined",
        "needs_clarification": "❓ Ambiguous — clarification requested",
        "abstained_low_score": "⚠️ Abstained (low retrieval confidence)",
    }.get(status, status)
    st.subheader(badge)
    if res["trace"].get("gate"):
        st.caption(f"Query gate → {res['trace']['gate']}")

    st.markdown(res["answer"])

    if status == "answered":
        verdict = res.get("verdict")
        vbadge = ("🟢 grounded — every quote verified verbatim in the retrieved context"
                  if verdict == "PASS" else f"🔴 {verdict} — quote not found in context")
        st.markdown(f"**Faithfulness:** {vbadge}")
        if res.get("abstained"):
            st.info("Model returned NOT_FOUND — no supporting clause in the retrieved excerpts.")
        if res.get("quotes"):
            st.markdown("**Verified quotes**")
            for q in res["quotes"]:
                st.markdown(f"> {q}")

        with st.expander("Retrieved chunks (hybrid-RRF scores)"):
            chunks = {c.chunk_id: c for c in lens.index.stores[contract].chunks}
            for cid, score in res["trace"].get("retrieved", []):
                ch = chunks.get(cid)
                snippet = (ch.text[:320] + "…") if ch else "(missing)"
                st.markdown(f"`{score:.3f}`  {snippet}")
