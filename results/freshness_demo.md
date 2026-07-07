# Knowledge-freshness demo — incremental add without re-embedding

- Base index: 3 contracts, 133 chunks, built in 18.2s
- **Added 1 new contract in 0.8s** (7 new chunks embedded)
- Existing contracts' vectors untouched (object-identity check): **True**
- New contract immediately queryable — top BM25 hit for "What is the governing law?":
  > NOW, THEREFORE, in consideration of the above premises and the mutual promises contained herein, as well as other good and valuable considerations, the receipt and sufficiency of w…

Re-embedding is only required when the embedding model version changes — not when the corpus grows.