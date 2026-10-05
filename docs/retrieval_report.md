# Retrieval evaluation

Дата: 2026-09-29.

Проверка команды 7: dense top-k baseline, reranker switch, exact wallet filter before ranking, fixed evaluation set and reindex after embedding model change.

## Index

- Embedding model: `hashing_vectorizer_word_1_2_dim384_v1`.
- Index version: `idx_422d98707f81`.
- Reindexed model: `hashing_vectorizer_word_1_2_dim512_v2`.
- Reindexed version: `idx_b2e139ea4a89`.

## Metrics

### dense_top_k

- Recall@3: `1.0`.
- MRR: `0.6944444444444443`.
- Foreign wallet errors: `0`.
- Avg latency ms: `0.6609750028777247`.
- Max latency ms: `1.278800016734749`.

### dense_top_k_reranker

- Recall@3: `1.0`.
- MRR: `0.7916666666666666`.
- Foreign wallet errors: `0`.
- Avg latency ms: `0.9850000060396269`.
- Max latency ms: `1.765700028045103`.

## Notes

- Address filter is applied before ranking for wallet queries.
- General methodology documents are searched separately with `include_general`.
- Qdrant points were exported to `data/vector_index/qdrant_points.jsonl`; local Qdrant collection creation is pending because `qdrant-client` is not importable in this Python environment.
- LLM generation is not connected in this command.
