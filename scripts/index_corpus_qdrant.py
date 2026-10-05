import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from src.retrieval import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_PATH,
    DEFAULT_QDRANT_POINTS_PATH,
    EMBEDDING_MODEL_V1,
    build_local_index,
    write_qdrant_points,
)


STATUS_PATH = PROJECT_ROOT / "results" / "qdrant_index_status.json"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_local"


def parse_args():
    parser = argparse.ArgumentParser(description="Index corpus chunks for retrieval/Qdrant.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_V1)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def try_qdrant_index(index, collection_name, recreate=False):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("qdrant"):
            return {
                "status": "pending_missing_qdrant_client",
                "reason": "qdrant-client is not importable in this Python environment",
            }
        raise

    if recreate and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=index["embedding_dim"], distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=record["point_id"], vector=record["vector"], payload=record["payload"])
        for record in index["records"]
    ]
    client.upsert(collection_name=collection_name, points=points)
    return {
        "status": "indexed",
        "collection_name": collection_name,
        "path": str(QDRANT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "points": len(points),
    }


def main():
    args = parse_args()
    index = build_local_index(output_path=DEFAULT_INDEX_PATH, embedding_model=args.embedding_model)
    write_qdrant_points(index, DEFAULT_QDRANT_POINTS_PATH)
    qdrant_result = try_qdrant_index(index, args.collection, recreate=args.recreate)
    status = {
        "collection_name": args.collection,
        "embedding_model": index["embedding_model"],
        "embedding_dim": index["embedding_dim"],
        "index_version": index["index_version"],
        "corpus_version": index["corpus_version"],
        "local_index_path": str(DEFAULT_INDEX_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "qdrant_points_path": str(DEFAULT_QDRANT_POINTS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "records": len(index["records"]),
        "qdrant": qdrant_result,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
