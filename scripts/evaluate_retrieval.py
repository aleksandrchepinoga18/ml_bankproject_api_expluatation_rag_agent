import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    EMBEDDING_MODEL_V1,
    EMBEDDING_MODEL_V2,
    LocalDenseRetriever,
    build_local_index,
    load_manifest,
)


EVAL_PATH = PROJECT_ROOT / "data" / "retrieval_eval.json"
RESULT_PATH = PROJECT_ROOT / "results" / "retrieval_eval.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "retrieval_report.md"


def build_eval_set():
    manifest = load_manifest()
    derived = [doc for doc in manifest["documents"] if doc["document_type"] == "observation_card"]
    synthetic = [doc for doc in manifest["documents"] if doc["document_type"] == "analyst_note"]
    method = [doc for doc in manifest["documents"] if doc["document_type"] == "methodology"]
    queries = []

    for doc in derived[:6]:
        queries.append(
            {
                "query_id": f"wallet_card_{doc['document_id']}",
                "query": f"What fields are available for wallet {doc['wallet_address']}?",
                "wallet_address": doc["wallet_address"],
                "scope": "wallet",
                "relevant_chunk_ids": [doc["chunk_ids"][0]],
            }
        )

    for doc in synthetic[:3]:
        queries.append(
            {
                "query_id": f"synthetic_note_{doc['document_id']}",
                "query": f"What demo analyst questions are listed for wallet {doc['wallet_address']}?",
                "wallet_address": doc["wallet_address"],
                "scope": "wallet",
                "relevant_chunk_ids": doc["chunk_ids"],
            }
        )

    for doc in method:
        queries.append(
            {
                "query_id": f"general_{doc['document_id']}",
                "query": doc["title"],
                "wallet_address": None,
                "scope": "general",
                "relevant_chunk_ids": doc["chunk_ids"],
            }
        )

    EVAL_PATH.write_text(json.dumps({"queries": queries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return queries


def load_or_build_eval_set():
    if not EVAL_PATH.exists():
        return build_eval_set()
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["queries"]


def reciprocal_rank(hits, relevant):
    for hit in hits:
        if hit.chunk_id in relevant:
            return 1.0 / hit.rank
    return 0.0


def evaluate_variant(retriever, queries, *, rerank, top_k):
    recalls = []
    reciprocal_ranks = []
    foreign_errors = 0
    latencies = []
    examples = []

    for query in queries:
        started = time.perf_counter()
        hits = retriever.search(
            query["query"],
            wallet_address=query["wallet_address"],
            include_general=query["scope"] == "general",
            top_k=top_k,
            rerank=rerank,
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
                if hit.wallet_address not in {query["wallet_address"], None}:
                    foreign_errors += 1

        examples.append(
            {
                "query_id": query["query_id"],
                "top_chunks": [hit.chunk_id for hit in hits],
                "relevant_chunk_ids": query["relevant_chunk_ids"],
                "recall": recall,
            }
        )

    return {
        "rerank": rerank,
        "top_k": top_k,
        "query_count": len(queries),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "foreign_wallet_errors": foreign_errors,
        "latency_ms_avg": sum(latencies) / len(latencies),
        "latency_ms_max": max(latencies),
        "examples": examples[:5],
    }


def write_report(result):
    lines = [
        "# Retrieval evaluation",
        "",
        "Дата: 2026-09-29.",
        "",
        "Проверка команды 7: dense top-k baseline, reranker switch, exact wallet filter "
        "before ranking, fixed evaluation set and reindex after embedding model change.",
        "",
        "## Index",
        "",
        f"- Embedding model: `{result['embedding_model']}`.",
        f"- Index version: `{result['index_version']}`.",
        f"- Reindexed model: `{result['reindex']['embedding_model']}`.",
        f"- Reindexed version: `{result['reindex']['index_version']}`.",
        "",
        "## Metrics",
        "",
    ]
    for variant_name, metrics in result["variants"].items():
        lines.extend(
            [
                f"### {variant_name}",
                "",
                f"- Recall@{metrics['top_k']}: `{metrics['recall_at_k']}`.",
                f"- MRR: `{metrics['mrr']}`.",
                f"- Foreign wallet errors: `{metrics['foreign_wallet_errors']}`.",
                f"- Avg latency ms: `{metrics['latency_ms_avg']}`.",
                f"- Max latency ms: `{metrics['latency_ms_max']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Address filter is applied before ranking for wallet queries.",
            "- General methodology documents are searched separately with `include_general`.",
            "- Qdrant points were exported to `data/vector_index/qdrant_points.jsonl`; local Qdrant collection creation is pending because `qdrant-client` is not importable in this Python environment.",
            "- LLM generation is not connected in this command.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    queries = load_or_build_eval_set()
    index = build_local_index(output_path=DEFAULT_INDEX_PATH, embedding_model=EMBEDDING_MODEL_V1)
    retriever = LocalDenseRetriever(DEFAULT_INDEX_PATH)
    baseline = evaluate_variant(retriever, queries, rerank=False, top_k=args.top_k)
    reranked = evaluate_variant(retriever, queries, rerank=True, top_k=args.top_k)
    reindexed = build_local_index(
        output_path=PROJECT_ROOT / "data" / "vector_index" / "local_dense_index_reindexed.json",
        embedding_model=EMBEDDING_MODEL_V2,
    )

    result = {
        "embedding_model": index["embedding_model"],
        "index_version": index["index_version"],
        "query_count": len(queries),
        "variants": {
            "dense_top_k": baseline,
            "dense_top_k_reranker": reranked,
        },
        "reindex": {
            "embedding_model": reindexed["embedding_model"],
            "index_version": reindexed["index_version"],
            "records": len(reindexed["records"]),
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
