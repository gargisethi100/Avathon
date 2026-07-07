"""
conversation.py — multi-turn conversational memory for ClauseLens.

History-aware retrieval via **condense-question**: a follow-up is rewritten into a
STANDALONE question using the running summary + recent turns, so *retrieval* works
(a pronoun-laden follow-up like "and its notice period?" otherwise retrieves the
wrong chunks — and the query gate would flag it AMBIGUOUS). The rewritten query then
flows through the unchanged gate → retrieve → generate → verify pipeline, so
grounding, citations, and the verifier are untouched: **memory only resolves
references, it never adds knowledge.**

Memory is bounded (last-N recent turns kept verbatim + a running summary of older
turns) and **contract-scoped** (reset on contract switch → no cross-document leakage).
"""
from __future__ import annotations

_CONTEXTUALIZE_SYSTEM = (
    "You rewrite a follow-up question in a contract-analysis chat into a fully "
    "STANDALONE question, resolving pronouns and references using the conversation "
    "summary and recent turns. If the question is already standalone, return it "
    "unchanged. Do NOT answer it. Output ONLY the rewritten question, nothing else."
)

_SUMMARIZE_SYSTEM = (
    "You maintain a concise running summary of a contract Q&A conversation. Update the "
    "prior summary with the older turns provided, in 3-4 sentences, keeping the clauses, "
    "parties, dates, and entities discussed so later references resolve. Output ONLY the "
    "updated summary."
)


def _converse(client, model_id, system, user, max_tokens):
    resp = client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()


def _format_turns(turns) -> str:
    return "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in turns)


def contextualize_query(client, model_id, summary, recent_turns, question):
    """Rewrite a follow-up into a standalone question. First turn -> no LLM call."""
    if not summary and not recent_turns:
        return question
    parts = []
    if summary:
        parts.append(f"CONVERSATION SUMMARY:\n{summary}")
    if recent_turns:
        parts.append(f"RECENT TURNS:\n{_format_turns(recent_turns)}")
    parts.append(f"FOLLOW-UP QUESTION: {question}")
    return _converse(client, model_id, _CONTEXTUALIZE_SYSTEM, "\n\n".join(parts), 120)


def summarize_history(client, model_id, prior_summary, folded_turns):
    user = (
        f"PRIOR SUMMARY:\n{prior_summary or '(none)'}\n\n"
        f"OLDER TURNS TO FOLD IN:\n{_format_turns(folded_turns)}"
    )
    return _converse(client, model_id, _SUMMARIZE_SYSTEM, user, 200)


class ClauseLensSession:
    """Stateful, per-conversation wrapper over a (shared, stateless) ClauseLens."""

    def __init__(self, lens, contract_id: str | None = None, max_recent: int = 5):
        self.lens = lens
        self.contract_id = contract_id
        self.max_recent = max_recent
        self.recent: list[tuple[str, str]] = []  # (question, answer_text)
        self.summary: str = ""
        self._client = lens.generator.client
        self._model = lens.generator.model_id

    def reset(self) -> None:
        self.recent = []
        self.summary = ""

    def set_contract(self, contract_id: str) -> None:
        if contract_id != self.contract_id:
            self.contract_id = contract_id
            self.reset()  # contract-scoped memory — no cross-document leakage

    def ask(self, question: str, contract_id: str | None = None) -> dict:
        if contract_id:
            self.set_contract(contract_id)
        standalone = contextualize_query(
            self._client, self._model, self.summary, self.recent, question
        )
        res = self.lens.answer(standalone, self.contract_id)
        res["standalone_question"] = standalone
        res["was_rewritten"] = standalone.strip() != question.strip()
        res["summary"] = self.summary

        # update bounded memory
        self.recent.append((question, res.get("answer", "")))
        if len(self.recent) > self.max_recent:
            overflow = self.recent[: -self.max_recent]
            self.recent = self.recent[-self.max_recent :]
            self.summary = summarize_history(self._client, self._model, self.summary, overflow)
        return res
