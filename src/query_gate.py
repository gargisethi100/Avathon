"""
query_gate.py — classify a query BEFORE retrieval (write-up: query analysis).

Routes each question into one of three lanes so the system fails gracefully
instead of forcing an answer:
  - IN_SCOPE      : a specific, contract-answerable question  -> proceed to retrieve
  - OUT_OF_SCOPE  : not about a contract at all                -> graceful decline
  - AMBIGUOUS     : contract-shaped but too vague to answer    -> ask to clarify

A tiny Claude classification (max 10 tokens) on the query alone — cheap, and it
doesn't need the index. The pipeline adds a retrieval-score backstop as a second
line of defense (verifier.should_abstain).
"""
from __future__ import annotations

import os

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

from config import BEDROCK_MODEL_ID, load_env

GATE_PROMPT = (
    "You route questions for a commercial-contract analysis assistant. "
    "Classify the user's question into EXACTLY one label:\n"
    "- IN_SCOPE: a specific question answerable from a contract (termination, "
    "governing law, indemnification, payment, dates, parties, liability, etc.).\n"
    "- OUT_OF_SCOPE: not about a contract at all (general knowledge, coding, "
    "weather, sports, etc.).\n"
    "- AMBIGUOUS: about contracts but too vague to answer precisely without "
    "clarification (e.g., 'what are the terms?', 'is this a good contract?').\n"
    "Respond with ONLY the label word."
)
LABELS = ("OUT_OF_SCOPE", "AMBIGUOUS", "IN_SCOPE")


class QueryGate:
    def __init__(self, model_id: str = BEDROCK_MODEL_ID, region: str | None = None):
        import boto3

        load_env()
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def classify(self, query: str) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": GATE_PROMPT}],
            messages=[{"role": "user", "content": [{"text": query}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0},
        )
        out = resp["output"]["message"]["content"][0]["text"].strip().upper()
        for label in LABELS:  # check OOS/AMBIGUOUS before IN_SCOPE (substring safety)
            if label in out:
                return label
        return "IN_SCOPE"
