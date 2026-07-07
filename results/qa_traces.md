# ClauseLens — Annotated Q&A Traces

Representative end-to-end traces (winner config: recursive-512 + bge-small + hybrid-RRF →
Claude Haiku 4.5 → deterministic verifier). Each shows the outcome, the generated answer,
and a **faithfulness annotation**. Successes *and* honest failures are included.

---

### ✅ 1. Answered — grounded (clean success)
- **Q:** parts related to **"Uncapped Liability"** to review?
- **Gate:** IN_SCOPE · **Verdict:** PASS · **token-F1:** 1.0
- **Answer:** *"…there IS a provision related to uncapped liability… **"NOTWITHSTANDING THE FOREGOING, NOTHING IN THIS SECTION IS INTENDED TO LIMIT OR RESTRICT THE DAMAGES AVAILABLE FOR BREACHES OF SECTION 3.1…"** [n]"*
- **Faithfulness:** the quoted carve-out is a verbatim substring of the retrieved context and exactly matches the gold span. ✔

### ✅ 2. Answered — grounded (partial-match success)
- **Q:** parts related to **"Anti-Assignment"** to review?
- **Gate:** IN_SCOPE · **Verdict:** PASS · **token-F1:** 0.78
- **Answer:** *"Yes, consent is required for assignment… **"Without the prior written consent of the other Party, such consent not to be unreasonably withheld…, neither Party shall sell…"**"*
- **Faithfulness:** quote verified verbatim; F1<1 only because the answer quotes the operative sentence while the gold span is longer. ✔

### ⛔ 3. Out-of-scope — gate declines (no retrieval/generation)
- **Q:** *"What is the capital of France?"*
- **Gate:** **OUT_OF_SCOPE** → *"That question isn't about this contract, so I can't answer it from the document."*
- **Faithfulness:** N/A — refused before retrieval, zero hallucination surface. ✔

### ❓ 4. Ambiguous — gate requests clarification
- **Q:** *"What are the terms?"*
- **Gate:** **AMBIGUOUS** → *"That's ambiguous for this contract — which clause or detail do you mean (which date, which party, which type of termination)?"*
- **Faithfulness:** N/A — clarification requested rather than guessing. ✔

### ⚠️ 5. Correct abstention (unanswerable clause)
- **Q:** parts related to a clause category **absent** from this contract
- **Gate:** IN_SCOPE · **Answer:** **NOT_FOUND** · **Verdict:** PASS (abstained)
- **Faithfulness:** the model declined rather than fabricate a clause that isn't present — the core safety behavior (abstention recall 0.83 across the sample). ✔

### 🔴 6. Honest failure — *unfaithful quote* (verifier strictness)
- **Q:** parts related to **"Termination For Convenience"**?
- **Verdict:** **UNGROUNDED** · **token-F1:** 0.83 (primary quote is correct!)
- **Answer:** *"…"XFN may terminate this Agreement by giving thirty (30) days written notice to the CEIS." [1] … only a 30-day notice period."*
- **Faithfulness annotation:** the *primary* quote is verbatim and correct, but the answer's secondary embellishment isn't a verbatim substring, so the **strict all-quotes gate flags the whole answer**. This is the verifier being conservative — a hard lower bound on faithfulness (0.53). Fix: verify only the primary citation / suppress embellishment.

### 🔴 7. Honest failure — *wrongful abstention* (interpretation, not retrieval)
- **Q:** parts related to **"Document Name"**? (gold answer = the contract title, e.g. *"SPONSORSHIP AGREEMENT"*)
- **Verdict:** PASS (abstained) · **but should have answered**
- **Answer:** **NOT_FOUND** — *"…refers to the contract as 'SPONSORSHIP AGREEMENT' in the header [5], but there is no dedicated section labelled Document Name."*
- **Faithfulness annotation:** the title *was* retrieved; the model just declined to treat "the contract is called X" as the answer. A question-interpretation gap, not a retrieval miss — a meaningful slice of the 46 wrongful-abstentions.
