import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_retrieval import build_eval_set  # noqa: E402
from src.retrieval import DEFAULT_INDEX_PATH, EMBEDDING_MODEL_V1, LocalDenseRetriever, build_local_index  # noqa: E402


EVAL_PATH = PROJECT_ROOT / "data" / "retrieval_eval.json"
RESULT_PATH = PROJECT_ROOT / "results" / "retrieval_search_comparison.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "retrieval_search_comparison.md"


def load_fixed_eval_set():
    if not EVAL_PATH.exists():
        return build_eval_set()
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["queries"]


def reciprocal_rank(hits, relevant):
    for hit in hits:
        if hit.chunk_id in relevant:
            return 1.0 / hit.rank
    return 0.0


def evaluate_variant(retriever, queries, *, variant_name, search_method, top_k):
    recalls = []
    reciprocal_ranks = []
    foreign_errors = 0
    latencies = []
    examples = []

    for query in queries:
        started = time.perf_counter()
        hits = search_method(
            query["query"],
            wallet_address=query["wallet_address"],
            include_general=query["scope"] == "general",
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        relevant = set(query["relevant_chunk_ids"])
        found = {hit.chunk_id for hit in hits}
        recall = len(relevant & found) / len(relevant)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank(hits, relevant))
        latencies.append(elapsed_ms)

        if query["wallet_address"]:
            for hit in hits:
                if hit.wallet_address != query["wallet_address"]:
                    foreign_errors += 1

        examples.append(
            {
                "query_id": query["query_id"],
                "top_chunks": [hit.chunk_id for hit in hits],
                "relevant_chunk_ids": query["relevant_chunk_ids"],
                "recall": recall,
                "reciprocal_rank": reciprocal_rank(hits, relevant),
            }
        )

    return {
        "variant": variant_name,
        "top_k": top_k,
        "query_count": len(queries),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "foreign_wallet_errors": foreign_errors,
        "latency_ms_avg": sum(latencies) / len(latencies),
        "latency_ms_max": max(latencies),
        "examples": examples[:5],
    }


def choose_primary_variant(variants):
    eligible = [item for item in variants.values() if item["foreign_wallet_errors"] == 0]
    candidates = eligible or list(variants.values())
    return max(
        candidates,
        key=lambda item: (
            item["recall_at_k"],
            item["mrr"],
            -item["foreign_wallet_errors"],
            -item["latency_ms_avg"],
        ),
    )


def write_report(result):
    top_k = result["top_k"]
    lines = [
        "# Retrieval search variant comparison",
        "",
        "Дата: 2026-09-30.",
        "",
        "Проверка команды 7а: сравнение точного wallet-фильтра + vector, wallet-фильтра + BM25, "
        "объединения vector+BM25 с удалением дублей и объединения + reranker на фиксированном eval-set.",
        "",
        "## Dataset",
        "",
        f"- Queries: `{result['query_count']}`.",
        f"- Embedding model: `{result['embedding_model']}`.",
        f"- Index version: `{result['index_version']}`.",
        "- Address filter is applied before ranking in every address-specific variant.",
        "",
        "## Metrics",
        "",
        "| Variant | Recall@{top_k} | MRR | Foreign wallet errors | Avg latency ms | Max latency ms |".format(
            top_k=top_k
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["variants"].items():
        lines.append(
            "| {name} | {recall:.6f} | {mrr:.6f} | {foreign} | {avg:.6f} | {max_latency:.6f} |".format(
                name=name,
                recall=metrics["recall_at_k"],
                mrr=metrics["mrr"],
                foreign=metrics["foreign_wallet_errors"],
                avg=metrics["latency_ms_avg"],
                max_latency=metrics["latency_ms_max"],
            )
        )

    primary = result["primary_variant"]
    primary_metrics = result["variants"][primary]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Primary retrieval path: `{primary}`.",
            f"- Reason: best Recall@{top_k}/MRR among variants with zero foreign-wallet errors. "
            f"Selected metrics: Recall@{top_k} `{primary_metrics['recall_at_k']}`, "
            f"MRR `{primary_metrics['mrr']}`, avg latency ms `{primary_metrics['latency_ms_avg']}`.",
            "",
            "## Notes",
            "",
            "- BM25 is implemented locally for reproducible offline comparison.",
            "- Hybrid uses reciprocal-rank fusion over vector and BM25 pools, then removes duplicate chunks.",
            "- Hybrid + reranker applies the same deterministic lexical reranker from command 7 after union.",
            "- LLM generation is not connected in this command.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    queries = load_fixed_eval_set()
    index = build_local_index(output_path=DEFAULT_INDEX_PATH, embedding_model=EMBEDDING_MODEL_V1)
    retriever = LocalDenseRetriever(DEFAULT_INDEX_PATH)

    variants = {
        "wallet_filter_vector": evaluate_variant(
            retriever,
            queries,
            variant_name="wallet_filter_vector",
            search_method=retriever.search_vector,
            top_k=args.top_k,
        ),
        "wallet_filter_bm25": evaluate_variant(
            retriever,
            queries,
            variant_name="wallet_filter_bm25",
            search_method=retriever.search_bm25,
            top_k=args.top_k,
        ),
        "wallet_filter_vector_bm25_union": evaluate_variant(
            retriever,
            queries,
            variant_name="wallet_filter_vector_bm25_union",
            search_method=retriever.search_hybrid,
            top_k=args.top_k,
        ),
        "wallet_filter_vector_bm25_union_reranker": evaluate_variant(
            retriever,
            queries,
            variant_name="wallet_filter_vector_bm25_union_reranker",
            search_method=lambda *args_, **kwargs: retriever.search_hybrid(*args_, rerank=True, **kwargs),
            top_k=args.top_k,
        ),
    }
    primary = choose_primary_variant(variants)["variant"]
    result = {
        "embedding_model": index["embedding_model"],
        "index_version": index["index_version"],
        "top_k": args.top_k,
        "query_count": len(queries),
        "variants": variants,
        "primary_variant": primary,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
