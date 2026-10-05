import json
import sys
from pathlib import Path
from typing import NoReturn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
INDEX_PATH = PROJECT_ROOT / "data" / "vector_index" / "local_dense_index.json"


def _fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path.relative_to(PROJECT_ROOT)}: {exc}")


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")

    rows = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _fail(f"invalid JSONL at {path.relative_to(PROJECT_ROOT)}:{line_no}: {exc}")
    return rows


def main() -> None:
    manifest = _load_json(CORPUS_DIR / "manifest.json")
    chunks = _load_jsonl(CORPUS_DIR / "chunks.jsonl")
    index = _load_json(INDEX_PATH)

    documents = manifest.get("documents") or []
    if manifest.get("document_count") != len(documents):
        _fail("manifest document_count does not match documents length")
    if manifest.get("chunk_count") != len(chunks):
        _fail("manifest chunk_count does not match chunks length")

    document_ids = {doc.get("document_id") for doc in documents}
    wallet_docs = [doc for doc in documents if doc.get("wallet_address")]
    methodology_docs = [doc for doc in documents if doc.get("document_type") == "methodology"]
    if not wallet_docs:
        _fail("corpus has no wallet-linked documents")
    if not methodology_docs:
        _fail("corpus has no methodology documents")

    chunk_ids = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            _fail("chunk without chunk_id")
        if chunk_id in chunk_ids:
            _fail(f"duplicate chunk_id: {chunk_id}")
        chunk_ids.add(chunk_id)
        if chunk.get("document_id") not in document_ids:
            _fail(f"chunk references unknown document: {chunk_id}")

    records = index.get("records") or []
    if not records:
        _fail("local dense index has no records")
    indexed_chunk_ids = {record.get("payload", {}).get("chunk_id") for record in records}
    if indexed_chunk_ids != chunk_ids:
        _fail("local dense index chunk IDs do not match corpus chunks")

    print(
        "Corpus check passed: "
        f"{len(documents)} documents, {len(chunks)} chunks, {len(records)} indexed records, "
        f"{len(methodology_docs)} methodology documents."
    )


if __name__ == "__main__":
    main()
