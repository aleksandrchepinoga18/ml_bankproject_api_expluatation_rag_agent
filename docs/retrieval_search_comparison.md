# Retrieval search variant comparison

Дата: 2026-09-30.

Проверка команды 7а: сравнение точного wallet-фильтра + vector, wallet-фильтра + BM25, объединения vector+BM25 с удалением дублей и объединения + reranker на фиксированном eval-set.

## Dataset

- Queries: `12`.
- Embedding model: `hashing_vectorizer_word_1_2_dim384_v1`.
- Index version: `idx_422d98707f81`.
- Address filter is applied before ranking in every address-specific variant.

## Metrics

| Variant | Recall@3 | MRR | Foreign wallet errors | Avg latency ms | Max latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallet_filter_vector | 1.000000 | 0.777778 | 0 | 0.645933 | 1.046100 |
| wallet_filter_bm25 | 1.000000 | 0.791667 | 0 | 0.057608 | 0.163600 |
| wallet_filter_vector_bm25_union | 1.000000 | 0.777778 | 0 | 0.923467 | 1.492000 |
| wallet_filter_vector_bm25_union_reranker | 1.000000 | 0.791667 | 0 | 1.155600 | 1.501700 |

## Decision

- Primary retrieval path: `wallet_filter_bm25`.
- Reason: best Recall@3/MRR among variants with zero foreign-wallet errors. Selected metrics: Recall@3 `1.0`, MRR `0.7916666666666666`, avg latency ms `0.057608337859467916`.

## Notes

- BM25 is implemented locally for reproducible offline comparison.
- Hybrid uses reciprocal-rank fusion over vector and BM25 pools, then removes duplicate chunks.
- Hybrid + reranker applies the same deterministic lexical reranker from command 7 after union.
- LLM generation is not connected in this command.
