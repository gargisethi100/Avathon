# End-to-end eval (300 QA: 150 answerable / 150 unanswerable)

- config: {'chunking': 'recursive', 'tokens': 512, 'embedder': 'bge-small', 'mode': 'hybrid'}
- answer token-F1 (all answerable): 0.4449
- answer token-F1 (when it answers): 0.5764
- quote-overlaps-gold rate: 0.62
- abstention precision / recall: 0.731 / 0.8333
- faithfulness (verifier PASS rate): 0.5271
- error buckets: {'correct_context_wrong_answer': 10, 'wrongful_answer': 25, 'wrongful_abstention': 46, 'unfaithful_quote': 46}