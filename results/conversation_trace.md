# ClauseLens — Conversational Memory Trace

A multi-turn thread over one contract (Centrack International / i-on interactive). Each
follow-up is rewritten into a **standalone** question (condense-question) using the running
summary + recent turns, then run through the *unchanged* gate → retrieve → generate → verify
pipeline. Memory = last-5 turns verbatim + a running summary; contract-scoped.

| # | User asks | Rewritten to (standalone) | Outcome |
|---|---|---|---|
| 1 | What is the termination provision? | *(already standalone — no rewrite)* | answered, cited |
| 2 | **And what is its notice period?** | **"What is the notice period required for termination under the termination provision in this contract?"** | answered — **pronoun "its" resolved** |
| 3 | Who are the parties? | "Who are the parties to this contract?" | *"Centrack International … and i-on interactive"* **[4]** · PASS |
| 4 | What is the governing law? | "…in the contract between Centrack International and i-on interactive?" | State of Florida · PASS |
| 5 | What is the agreement date? | "…between Centrack International and i-on interactive?" | *"the 6th day of April, 1999"* **[2]** |
| 6 | Is there a confidentiality clause? | "…between Centrack International and i-on interactive?" | **NOT_FOUND** — abstains · PASS |

**Observations**
- Turn 2 is the headline: the bare pronoun follow-up ("its") is rewritten so **retrieval**
  actually finds the notice-period clause — a bare "and its notice period?" would otherwise
  retrieve nothing and the gate would flag it AMBIGUOUS.
- Turns 4–6 show the memory *injecting the parties' names* into later queries automatically.
- After turn 6 the recent buffer is trimmed to 5 and the running summary is populated
  (Hosted-Site agreement, 6-month term, termination provisions …).
- **Grounding is untouched** — citations, verifier verdicts (PASS), and correct abstention
  (turn 6) all still hold. Memory only resolves references; it never adds knowledge.
