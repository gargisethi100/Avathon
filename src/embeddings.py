"""
embeddings.py — pluggable embedding backends behind one interface.

The embedding model is a *measured* axis (Q16), so all backends expose the same
API and are swapped by name:

  - "bge-small" / "bge-base" : local, CPU, free (sentence-transformers).
  - "bedrock-cohere"         : Cohere Embed English v3 via AWS Bedrock (the cloud
                               comparison — "does a free local model suffice?").

Retrieval asymmetry: bge (and Cohere) want a *query* instruction/type distinct
from *passages*, so we expose encode_queries() vs encode_passages(). All vectors
are L2-normalized, so a FAISS inner-product index == cosine similarity.
"""
from __future__ import annotations

import os
from typing import Protocol

import numpy as np

# HF model downloads go through the corporate SSL-inspection proxy.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _l2norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class Embedder(Protocol):
    name: str
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class BGEEmbedder:
    """Local sentence-transformers bge-*-en-v1.5 (unit-normalized)."""

    def __init__(self, model_id: str, name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)
        self.name = name
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        inp = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            inp,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, BGE_QUERY_PREFIX)


class BedrockCohereEmbedder:
    """Cohere Embed English v3 via AWS Bedrock (1024-dim). Lazy client."""

    def __init__(self, name: str = "bedrock-cohere",
                 model_id: str = "cohere.embed-english-v3", region: str | None = None):
        import boto3

        self.name = name
        self.model_id = model_id
        self.dim = 1024
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        import json

        out: list[list[float]] = []
        for i in range(0, len(texts), 96):  # Cohere caps at 96 texts/call
            batch = texts[i : i + 96]
            resp = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"texts": batch, "input_type": input_type, "truncate": "END"}),
            )
            out.extend(json.loads(resp["body"].read())["embeddings"])
        return _l2norm(np.asarray(out, dtype="float32"))

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "search_document")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "search_query")


_LOCAL = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}


def get_embedder(name: str) -> Embedder:
    if name in _LOCAL:
        return BGEEmbedder(_LOCAL[name], name)
    if name == "bedrock-cohere":
        return BedrockCohereEmbedder()
    raise ValueError(f"unknown embedder: {name!r} (choices: {list(_LOCAL) + ['bedrock-cohere']})")


EMBEDDERS = tuple(_LOCAL) + ("bedrock-cohere",)
