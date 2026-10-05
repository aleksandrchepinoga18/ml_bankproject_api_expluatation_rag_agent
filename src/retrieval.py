import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
VECTOR_INDEX_DIR = PROJECT_ROOT / "data" / "vector_index"
DEFAULT_INDEX_PATH = VECTOR_INDEX_DIR / "local_dense_index.json"
DEFAULT_QDRANT_POINTS_PATH = VECTOR_INDEX_DIR / "qdrant_points.jsonl"
DEFAULT_COLLECTION_NAME = "wallet_risk_demo_corpus"

EMBEDDING_MODEL_V1 = "hashing_vectorizer_word_1_2_dim384_v1"
EMBEDDING_MODEL_V2 = "hashing_vectorizer_word_1_2_dim512_v2"


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    score: float
    wallet_address: str | None
    source_kind: str
    text: str
    rank: int
    document_type: str | None = None


def embedding_dim(embedding_model: str) -> int:
    if "dim512" in embedding_model:
        return 512
    return 384


def index_version(corpus_version: str, embedding_model: str) -> str:
    payload = f"{corpus_version}:{embedding_model}"
    return "idx_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _vectorizer(embedding_model: str):
    return HashingVectorizer(
        n_features=embedding_dim(embedding_model),
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )


def embed_texts(texts: list[str], embedding_model: str) -> np.ndarray:
    matrix = _vectorizer(embedding_model).transform(texts)
    return matrix.toarray().astype(float)


def load_manifest(corpus_dir: Path = CORPUS_DIR) -> dict[str, Any]:
    with (corpus_dir / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    with (corpus_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _document_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {doc["document_id"]: doc for doc in manifest["documents"]}


def payload_for_chunk(
    chunk: dict[str, Any],
    document: dict[str, Any],
    embedding_model: str,
    idx_version: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "chunk_index": chunk["chunk_index"],
        "wallet_address": document["wallet_address"],
        "source_kind": document["source_kind"],
        "document_type": document["document_type"],
        "source_row_ids": document["source_row_ids"],
        "corpus_version": document["corpus_version"],
        "embedding_version": embedding_model,
        "index_version": idx_version,
        "text_hash": chunk["text_hash"],
        "text": chunk["text"],
    }


def build_local_index(
    corpus_dir: Path = CORPUS_DIR,
    output_path: Path = DEFAULT_INDEX_PATH,
    embedding_model: str = EMBEDDING_MODEL_V1,
) -> dict[str, Any]:
    manifest = load_manifest(corpus_dir)
    chunks = load_chunks(corpus_dir)
    documents = _document_lookup(manifest)
    idx_version = index_version(manifest["corpus_version"], embedding_model)
    vectors = embed_texts([chunk["text"] for chunk in chunks], embedding_model)

    records = []
    for idx, chunk in enumerate(chunks):
        document = documents[chunk["document_id"]]
        records.append(
            {
                "point_id": idx,
                "vector": vectors[idx].tolist(),
                "payload": payload_for_chunk(chunk, document, embedding_model, idx_version),
            }
        )

    index = {
        "collection_name": DEFAULT_COLLECTION_NAME,
        "corpus_version": manifest["corpus_version"],
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim(embedding_model),
        "index_version": idx_version,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def write_qdrant_points(index: dict[str, Any], output_path: Path = DEFAULT_QDRANT_POINTS_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in index["records"]:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_local_index(path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1.0
    return matrix.dot(query_vector) / denom


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in normalized.split() if token}


def _token_list(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in normalized.split() if token]


def rerank_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    query_tokens = _tokens(query)
    reranked = []
    for hit in hits:
        hit_tokens = _tokens(hit.text)
        overlap = len(query_tokens & hit_tokens)
        lexical = overlap / math.sqrt(max(len(hit_tokens), 1))
        reranked.append((hit.score + lexical, hit))
    reranked.sort(key=lambda item: item[0], reverse=True)
    return [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            score=float(score),
            wallet_address=hit.wallet_address,
            source_kind=hit.source_kind,
            text=hit.text,
            rank=rank,
        )
        for rank, (score, hit) in enumerate(reranked, 1)
    ]


class LocalDenseRetriever:
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH):
        self.index = load_local_index(index_path)
        self.records = self.index["records"]
        self.embedding_model = self.index["embedding_model"]
        self.matrix = np.array([record["vector"] for record in self.records], dtype=float)
        self._doc_tokens = [_token_list(record["payload"]["text"]) for record in self.records]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_length = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        self._bm25_doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            self._bm25_doc_freq.update(set(tokens))
        self.last_elapsed_ms = 0.0

    def _candidate_indices(self, wallet_address: str | None, include_general: bool) -> list[int]:
        candidates = []
        for idx, record in enumerate(self.records):
            payload_wallet = record["payload"].get("wallet_address")
            if wallet_address is not None:
                if payload_wallet != wallet_address:
                    continue
            elif include_general:
                if payload_wallet is not None:
                    continue
            elif payload_wallet is None:
                continue
            candidates.append(idx)
        return candidates

    def _hits_from_ranked(self, ranked: list[tuple[int, float]], top_k: int) -> list[RetrievalHit]:
        hits = []
        for rank, (idx, score) in enumerate(ranked[:top_k], 1):
            payload = self.records[idx]["payload"]
            hits.append(
                RetrievalHit(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    score=float(score),
                    wallet_address=payload.get("wallet_address"),
                    source_kind=payload["source_kind"],
                    text=payload["text"],
                    rank=rank,
                    document_type=payload.get("document_type"),
                )
            )
        return hits

    def _vector_ranked(
        self,
        query: str,
        *,
        wallet_address: str | None,
        include_general: bool,
    ) -> list[tuple[int, float]]:
        query_vector = embed_texts([query], self.embedding_model)[0]
        scores = _cosine_scores(query_vector, self.matrix)
        ranked = [(idx, float(scores[idx])) for idx in self._candidate_indices(wallet_address, include_general)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _bm25_ranked(
        self,
        query: str,
        *,
        wallet_address: str | None,
        include_general: bool,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[int, float]]:
        query_tokens = _token_list(query)
        query_terms = Counter(query_tokens)
        total_docs = len(self.records)
        ranked = []
        for idx in self._candidate_indices(wallet_address, include_general):
            doc_tokens = self._doc_tokens[idx]
            term_counts = Counter(doc_tokens)
            doc_len = max(self._doc_lengths[idx], 1)
            score = 0.0
            for term, query_count in query_terms.items():
                term_freq = term_counts.get(term, 0)
                if term_freq == 0:
                    continue
                doc_freq = self._bm25_doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                norm = term_freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_length, 1e-9))
                score += query_count * idf * (term_freq * (k1 + 1) / norm)
            ranked.append((idx, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def search_vector(
        self,
        query: str,
        *,
        wallet_address: str | None = None,
        include_general: bool = False,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[RetrievalHit]:
        started = time.perf_counter()
        hits = self._hits_from_ranked(
            self._vector_ranked(query, wallet_address=wallet_address, include_general=include_general),
            top_k=top_k,
        )
        if rerank:
            hits = rerank_hits(query, hits)[:top_k]
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000
        return hits

    def search_bm25(
        self,
        query: str,
        *,
        wallet_address: str | None = None,
        include_general: bool = False,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[RetrievalHit]:
        started = time.perf_counter()
        hits = self._hits_from_ranked(
            self._bm25_ranked(query, wallet_address=wallet_address, include_general=include_general),
            top_k=top_k,
        )
        if rerank:
            hits = rerank_hits(query, hits)[:top_k]
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000
        return hits

    def search_hybrid(
        self,
        query: str,
        *,
        wallet_address: str | None = None,
        include_general: bool = False,
        top_k: int = 5,
        rerank: bool = False,
        pool_k: int | None = None,
    ) -> list[RetrievalHit]:
        started = time.perf_counter()
        pool_k = pool_k or max(top_k * 2, top_k)
        vector_ranked = self._vector_ranked(query, wallet_address=wallet_address, include_general=include_general)
        bm25_ranked = self._bm25_ranked(query, wallet_address=wallet_address, include_general=include_general)

        fused: dict[int, float] = {}
        for ranked in (vector_ranked[:pool_k], bm25_ranked[:pool_k]):
            for rank, (idx, _score) in enumerate(ranked, 1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        hits = self._hits_from_ranked(ranked, top_k=len(ranked))
        if rerank:
            hits = rerank_hits(query, hits)
        hits = hits[:top_k]
        hits = [
            RetrievalHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=hit.score,
                wallet_address=hit.wallet_address,
                source_kind=hit.source_kind,
                text=hit.text,
                rank=rank,
            )
            for rank, hit in enumerate(hits, 1)
        ]
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000
        return hits

    def search(
        self,
        query: str,
        *,
        wallet_address: str | None = None,
        include_general: bool = False,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[RetrievalHit]:
        return self.search_vector(
            query,
            wallet_address=wallet_address,
            include_general=include_general,
            top_k=top_k,
            rerank=rerank,
        )


def hits_to_dicts(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [asdict(hit) for hit in hits]
