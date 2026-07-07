# Retrieval OFAT ablations (244 answerable queries, 20-contract stratified subset, seed 42)

| axis | config | mode | P@5 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| retrieval | recursive-256/bge-small | dense | 0.1295 | 0.4686 | 0.6052 | 0.4166 | 0.4368 |
| retrieval | recursive-256/bge-small | bm25 | 0.1459 | 0.5433 | 0.6602 | 0.454 | 0.4814 |
| retrieval | recursive-256/bge-small | hybrid | 0.1492 | 0.546 | 0.6649 | 0.4674 | 0.4882 |
| retrieval | recursive-256/bge-small | hybrid_rerank | 0.1295 | 0.4819 | 0.6414 | 0.4153 | 0.4516 |
| chunking | recursive-256/bge-small | hybrid | 0.1492 | 0.546 | 0.6649 | 0.4674 | 0.4882 |
| chunking | recursive-512/bge-small | hybrid | 0.1525 | 0.5955 | 0.7531 | 0.4704 | 0.5221 |
| chunking | structure_aware-256/bge-small | hybrid | 0.1418 | 0.5452 | 0.6686 | 0.4515 | 0.4823 |
| chunking | structure_aware-512/bge-small | hybrid | 0.15 | 0.5995 | 0.7442 | 0.4635 | 0.5198 |
| embedding | recursive-256/bge-small | hybrid | 0.1492 | 0.546 | 0.6649 | 0.4674 | 0.4882 |
| embedding | recursive-256/bge-base | hybrid | 0.1484 | 0.5426 | 0.6561 | 0.48 | 0.4959 |