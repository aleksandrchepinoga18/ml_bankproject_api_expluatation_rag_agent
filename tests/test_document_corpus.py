import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.document_corpus import (
    CORPUS_VERSION,
    CREATED_AT,
    CorpusDocument,
    chunk_document,
    sha256_text,
    write_corpus,
)


def _load_chunks(path):
    with (path / "chunks.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_demo_corpus_generation_is_deterministic(tmp_path, synthetic_dataset_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = write_corpus(
        output_dir=first,
        dataset_path=synthetic_dataset_path,
        n_wallets=8,
        seed=42,
        rebuild=True,
    )
    manifest_b = write_corpus(
        output_dir=second,
        dataset_path=synthetic_dataset_path,
        n_wallets=8,
        seed=42,
        rebuild=True,
    )

    manifest_a_clean = {k: v for k, v in manifest_a.items() if k != "dataset_path"}
    manifest_b_clean = {k: v for k, v in manifest_b.items() if k != "dataset_path"}
    assert manifest_a_clean == manifest_b_clean
    assert (first / "chunks.jsonl").read_text(encoding="utf-8") == (
        second / "chunks.jsonl"
    ).read_text(encoding="utf-8")


def test_document_links_match_dataset_rows(tmp_path, synthetic_dataset_path: Path):
    output = tmp_path / "corpus"
    manifest = write_corpus(
        output_dir=output,
        dataset_path=synthetic_dataset_path,
        n_wallets=6,
        seed=42,
        rebuild=True,
    )
    dataset = pd.read_parquet(synthetic_dataset_path).reset_index(names="source_row_id")

    documents = manifest["documents"]
    derived_cards = [doc for doc in documents if doc["document_type"] == "observation_card"]
    assert derived_cards

    for doc in derived_cards:
        assert doc["source_kind"] == "derived_data"
        assert doc["wallet_address"]
        assert doc["source_row_ids"]
        row = dataset.loc[dataset["source_row_id"] == doc["source_row_ids"][0]].iloc[0]
        assert row["wallet_address"] == doc["wallet_address"]
        text = (output / doc["path"]).read_text(encoding="utf-8")
        assert "does not assert fraud" in text
        assert "score" not in text.lower()


def test_synthetic_documents_are_explicitly_marked(tmp_path, synthetic_dataset_path: Path):
    output = tmp_path / "corpus"
    manifest = write_corpus(
        output_dir=output,
        dataset_path=synthetic_dataset_path,
        n_wallets=6,
        seed=42,
        rebuild=True,
    )

    synthetic_docs = [
        doc for doc in manifest["documents"] if doc["source_kind"] == "synthetic_demo"
    ]
    assert synthetic_docs

    for doc in synthetic_docs:
        text = (output / doc["path"]).read_text(encoding="utf-8")
        assert "synthetic_demo" in text
        assert "not a real investigation" in text or "intentionally unrelated" in text


def test_general_methodology_documents_have_no_wallet_address(tmp_path, synthetic_dataset_path: Path):
    output = tmp_path / "corpus"
    manifest = write_corpus(
        output_dir=output,
        dataset_path=synthetic_dataset_path,
        n_wallets=6,
        seed=42,
        rebuild=True,
    )

    method_docs = [doc for doc in manifest["documents"] if doc["document_type"] == "methodology"]
    assert method_docs
    assert all(doc["wallet_address"] is None for doc in method_docs)
    assert all(doc["source_row_ids"] == [] for doc in method_docs)


def test_chunks_preserve_document_offsets_and_hashes(tmp_path, synthetic_dataset_path: Path):
    output = tmp_path / "corpus"
    manifest = write_corpus(
        output_dir=output,
        dataset_path=synthetic_dataset_path,
        n_wallets=6,
        seed=42,
        rebuild=True,
    )
    chunks = _load_chunks(output)
    document_by_id = {doc["document_id"]: doc for doc in manifest["documents"]}

    assert chunks
    for chunk in chunks:
        doc = document_by_id[chunk["document_id"]]
        text = (output / doc["path"]).read_text(encoding="utf-8")
        assert text[chunk["offset_start"] : chunk["offset_end"]] == chunk["text"]
        assert chunk["text_hash"] == sha256_text(chunk["text"])
        assert chunk["embedding_version"] == "not_indexed_v1"
        assert chunk["index_version"] == "not_indexed_v1"

    multi_chunk_docs: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        multi_chunk_docs.setdefault(chunk["document_id"], []).append(chunk)
    for doc_chunks in multi_chunk_docs.values():
        ordered = sorted(doc_chunks, key=lambda item: item["chunk_index"])
        if len(ordered) > 1:
            assert ordered[0]["next_chunk_id"] == ordered[1]["chunk_id"]
            assert ordered[1]["previous_chunk_id"] == ordered[0]["chunk_id"]


def test_chunk_boundary_preserves_neighbor_context_for_split_fact():
    boundary_text = (
        "# Boundary QA document\n\n"
        "source_kind: derived_data\n"
        "wallet_address: 0xboundary\n"
        "source_row_id: 1\n\n"
        "## Fact setup\n"
        + ("A" * 230)
        + "\n\n"
        "The boundary fact begins here: risky_tx_count is reported in the source row.\n\n"
        "The boundary fact continues here: the value must be interpreted with the row "
        "snapshot and not as a fraud finding.\n"
    )
    document = CorpusDocument(
        document_id="boundary_doc",
        document_type="qa_fixture",
        source_kind="derived_data",
        wallet_address="0xboundary",
        source_row_ids=[1],
        snapshot_time="2026-09-29T00:00:00Z",
        created_at=CREATED_AT,
        corpus_version=CORPUS_VERSION,
        path="documents/boundary_doc.md",
        text_hash=sha256_text(boundary_text),
        title="Boundary QA document",
        text=boundary_text,
    )

    chunks = chunk_document(document, max_chars=430)

    assert len(chunks) >= 2
    assert "The boundary fact begins here" in chunks[0].text
    assert "The boundary fact continues here" in chunks[1].text
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
