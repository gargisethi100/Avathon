"""
generation.py — grounded answer generation via Claude on Bedrock (converse API).

Prompt contract (layer 1 of the hallucination defense, D-11): answer ONLY from the
provided excerpts; quote the supporting text verbatim; cite the excerpt number;
output exactly NOT_FOUND when the excerpts don't contain the answer. The excerpts
are presented as [1], [2], ... (short, reliable for the model to cite) and mapped
back to real chunk_ids. The deterministic verifier (verifier.py) is layer 2.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

from config import BEDROCK_MODEL_ID, load_env

SYSTEM_PROMPT = (
    "You are ClauseLens, a contract-analysis assistant for legal teams. "
    "Answer the user's question using ONLY the provided contract excerpts.\n"
    "Rules:\n"
    "1. If the excerpts answer the question, respond concisely, quote the exact "
    "supporting text VERBATIM in double quotes, and cite the excerpt number in "
    "square brackets, e.g. [2].\n"
    "2. If the excerpts do NOT contain the answer, respond with exactly: NOT_FOUND\n"
    "3. Never use outside knowledge or infer beyond the excerpts."
)

_CITE_RE = re.compile(r"\[(\d+)\]")
_QUOTE_RE = re.compile(r"\"([^\"]+)\"")


@dataclass
class Answer:
    text: str
    cited_chunk_ids: list[str]
    quotes: list[str]
    abstained: bool
    context_map: dict = field(default_factory=dict)  # display_index -> chunk
    usage: dict = field(default_factory=dict)


def _format_context(chunks):
    """chunks: list of Chunk or (Chunk, score). Returns (context_str, {i: chunk})."""
    parts, mapping = [], {}
    for i, item in enumerate(chunks, 1):
        ch = item[0] if isinstance(item, tuple) else item
        mapping[i] = ch
        parts.append(f"[{i}] {ch.text}")
    return "\n\n".join(parts), mapping


class BedrockGenerator:
    def __init__(self, model_id: str = BEDROCK_MODEL_ID, region: str | None = None):
        import boto3

        load_env()
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def answer(self, question: str, chunks, max_tokens: int = 400) -> Answer:
        context, mapping = _format_context(chunks)
        user = f"CONTRACT EXCERPTS:\n\n{context}\n\nQUESTION: {question}"
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        abstained = text.strip().upper().startswith("NOT_FOUND")
        cited_idx = [int(n) for n in _CITE_RE.findall(text)]
        cited_chunk_ids = [mapping[i].chunk_id for i in cited_idx if i in mapping]
        quotes = [] if abstained else _QUOTE_RE.findall(text)
        return Answer(
            text=text,
            cited_chunk_ids=cited_chunk_ids,
            quotes=quotes,
            abstained=abstained,
            context_map=mapping,
            usage=resp.get("usage", {}),
        )
