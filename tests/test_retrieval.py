import json

from src.retrieval import EMBEDDING_MODEL_V1, LocalDenseRetriever, build_local_index


def test_local_dense_retrieval_filters_wallet_before_ranking(tmp_path):
    index_path = tmp_path / "index.json"
    index = build_local_index(output_path=index_path, embedding_model=EMBEDDING_MODEL_V1)
    doc = next(
        record["payload"]
        for record in index["records"]
        if record["payload"]["wallet_address"] is not None
    )
    retriever = LocalDenseRetriever(index_path)

    hits = retriever.search(
        "available row fields risky transaction count",
        wallet_address=doc["wallet_address"],
        top_k=5,
    )

    assert hits
    assert all(hit.wallet_address == doc["wallet_address"] for hit in hits)


def test_general_retrieval_excludes_wallet_documents_when_requested(tmp_path):
    index_path = tmp_path / "index.json"
    build_local_index(output_path=index_path, embedding_model=EMBEDDING_MODEL_V1)
    retriever = LocalDenseRetriever(index_path)

    hits = retriever.search(
        "Retrieval rules for wallet documents",
        wallet_address=None,
        include_general=True,
        top_k=5,
    )

    assert hits
    assert all(hit.wallet_address is None for hit in hits)
    assert any(hit.document_id == "method_retrieval_rules" for hit in hits)


def test_reranker_switch_preserves_result_contract(tmp_path):
    index_path = tmp_path / "index.json"
    build_local_index(output_path=index_path, embedding_model=EMBEDDING_MODEL_V1)
    retriever = LocalDenseRetriever(index_path)

    plain = retriever.search("scoring interpretation limits", include_general=True, top_k=3)
    reranked = retriever.search(
        "scoring interpretation limits",
        include_general=True,
        top_k=3,
        rerank=True,
    )

    assert len(plain) == len(reranked)
    assert {hit.chunk_id for hit in plain} == {hit.chunk_id for hit in reranked}
    assert [hit.rank for hit in reranked] == [1, 2, 3]


def test_bm25_and_hybrid_filter_wallet_before_ranking(tmp_path):
    index_path = tmp_path / "index.json"
    index = build_local_index(output_path=index_path, embedding_model=EMBEDDING_MODEL_V1)
    wallet_address = next(
        record["payload"]["wallet_address"]
        for record in index["records"]
        if record["payload"]["wallet_address"] is not None
    )
    retriever = LocalDenseRetriever(index_path)

    bm25_hits = retriever.search_bm25(
        f"risk score source rows for {wallet_address}",
        wallet_address=wallet_address,
        top_k=5,
    )
    hybrid_hits = retriever.search_hybrid(
        f"risk score source rows for {wallet_address}",
        wallet_address=wallet_address,
        top_k=5,
    )

    assert bm25_hits
    assert hybrid_hits
    assert all(hit.wallet_address == wallet_address for hit in bm25_hits)
    assert all(hit.wallet_address == wallet_address for hit in hybrid_hits)


def test_hybrid_union_removes_duplicate_chunks(tmp_path):
    index_path = tmp_path / "index.json"
    index = build_local_index(output_path=index_path, embedding_model=EMBEDDING_MODEL_V1)
    wallet_address = next(
        record["payload"]["wallet_address"]
        for record in index["records"]
        if record["payload"]["wallet_address"] is not None
    )
    retriever = LocalDenseRetriever(index_path)

    hits = retriever.search_hybrid(
        f"available fields wallet {wallet_address}",
        wallet_address=wallet_address,
        top_k=5,
        pool_k=5,
    )

    chunk_ids = [hit.chunk_id for hit in hits]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_eval_file_has_fixed_queries_after_evaluation_script_exists():
    eval_path = "data/retrieval_eval.json"
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    assert data["queries"]
    assert all("relevant_chunk_ids" in query for query in data["queries"])
