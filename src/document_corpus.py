import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "corpus"

CORPUS_VERSION = "demo_corpus_v1"
CREATED_AT = "2026-09-29T00:00:00Z"
EMBEDDING_VERSION = "not_indexed_v1"
INDEX_VERSION = "not_indexed_v1"
DEFAULT_SEED = 42

CARD_FIELDS = [
    "borrow_timestamp",
    "first_tx_timestamp",
    "last_tx_timestamp",
    "incoming_tx_count",
    "outgoing_tx_count",
    "risky_tx_count",
    "risky_unique_contract_count",
    "borrow_count",
    "repay_count",
    "deposit_count",
    "withdraw_amount_sum_eth",
    "total_balance_eth",
]


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    document_type: str
    source_kind: str
    wallet_address: str | None
    source_row_ids: list[int]
    snapshot_time: str | None
    created_at: str
    corpus_version: str
    path: str
    text_hash: str
    title: str
    text: str


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    offset_start: int
    offset_end: int
    text_hash: str
    embedding_version: str
    index_version: str
    previous_chunk_id: str | None
    next_chunk_id: str | None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def timestamp_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def select_source_rows(dataset_path: Path, n_wallets: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path).reset_index(names="source_row_id")
    first_rows = (
        df.sort_values(["wallet_address", "borrow_timestamp", "source_row_id"])
        .groupby("wallet_address", as_index=False)
        .first()
    )
    n = min(n_wallets, len(first_rows))
    sampled = first_rows.sample(n=n, random_state=seed).sort_values("wallet_address")
    return sampled.reset_index(drop=True)


def observation_card_text(row: pd.Series, document_id: str) -> str:
    wallet = row["wallet_address"]
    source_row_id = int(row["source_row_id"])
    snapshot_time = timestamp_to_iso(row.get("borrow_timestamp"))
    lines = [
        f"# Observation card {document_id}",
        "",
        "source_kind: derived_data",
        "This card is deterministically derived from fields present in the local dataset.",
        "It is not an independent investigation and does not assert fraud.",
        "",
        f"wallet_address: {wallet}",
        f"source_row_id: {source_row_id}",
        f"snapshot_time: {snapshot_time}",
        "",
        "## Available row fields",
    ]
    for field in CARD_FIELDS:
        if field not in row.index:
            continue
        value = clean_value(row[field])
        if value is None:
            continue
        if field.endswith("_timestamp"):
            lines.append(f"- {field}: {value} ({timestamp_to_iso(value)})")
        else:
            lines.append(f"- {field}: {value}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "These values describe one row-level observation. Counts and amounts are dataset fields.",
            "They are not legal findings, complaints, or confirmed blockchain investigations.",
        ]
    )
    return "\n".join(lines) + "\n"


def synthetic_note_text(row: pd.Series, document_id: str) -> str:
    wallet = row["wallet_address"]
    source_row_id = int(row["source_row_id"])
    snapshot_time = timestamp_to_iso(row.get("borrow_timestamp"))
    return (
        f"# Synthetic demo analyst note {document_id}\n\n"
        "source_kind: synthetic_demo\n"
        "This note is fictional and exists only to test retrieval behavior. "
        "It is not a real investigation, complaint, transaction record, or legal conclusion.\n\n"
        f"wallet_address: {wallet}\n"
        f"related_source_row_id: {source_row_id}\n"
        f"snapshot_time: {snapshot_time}\n\n"
        "## Demo hypothesis\n"
        "For a portfolio demo, an analyst might ask whether activity counts, risky transaction "
        "counts, and lending activity should be reviewed together. The note intentionally "
        "uses cautious language and does not claim that the wallet committed fraud.\n\n"
        "## Questions to check\n"
        "- Are the cited row fields available at the stated snapshot time?\n"
        "- Does the model explanation separate score from evidence?\n"
        "- Are documents for other wallet addresses excluded before retrieval?\n"
    )


def methodology_documents() -> list[tuple[str, str, str]]:
    return [
        (
            "method_scoring_limits",
            "methodology",
            "# Scoring interpretation limits\n\n"
            "source_kind: derived_data\n"
            "A model score is an output of a saved LightGBM artifact for a row-level observation. "
            "It must not be described as a calibrated real-world probability unless calibration "
            "has been separately verified.\n\n"
            "Document claims about a specific wallet require a linked wallet_address and source "
            "row IDs. General methodology documents do not establish facts about any wallet.\n",
        ),
        (
            "method_feature_dictionary",
            "methodology",
            "# Feature dictionary notes\n\n"
            "source_kind: derived_data\n"
            "Fields ending in _count are dataset counts. Fields ending in _timestamp are Unix-like "
            "timestamps in the source table. Aggregate windows and availability at prediction time "
            "must be documented before production use.\n\n"
            "Temporal leakage review remains open for timestamp and aggregate features.\n",
        ),
        (
            "method_retrieval_rules",
            "methodology",
            "# Retrieval rules for wallet documents\n\n"
            "source_kind: derived_data\n"
            "For an address-specific question, exact wallet_address linkage must be checked before "
            "semantic ranking. Semantic similarity alone does not prove that a document belongs to "
            "a wallet. Documents with no wallet_address are general context only.\n",
        ),
        (
            "irrelevant_demo_text",
            "irrelevant",
            "# Irrelevant demo text\n\n"
            "source_kind: synthetic_demo\n"
            "This synthetic_demo document is intentionally unrelated to wallet risk. It mentions "
            "a generic documentation workflow and should not be retrieved as wallet evidence.\n",
        ),
    ]


def make_document(
    document_id: str,
    document_type: str,
    source_kind: str,
    wallet_address: str | None,
    source_row_ids: list[int],
    snapshot_time: str | None,
    text: str,
) -> CorpusDocument:
    path = f"documents/{document_id}.md"
    title = text.splitlines()[0].lstrip("# ").strip()
    return CorpusDocument(
        document_id=document_id,
        document_type=document_type,
        source_kind=source_kind,
        wallet_address=wallet_address,
        source_row_ids=source_row_ids,
        snapshot_time=snapshot_time,
        created_at=CREATED_AT,
        corpus_version=CORPUS_VERSION,
        path=path,
        text_hash=sha256_text(text),
        title=title,
        text=text,
    )


def generate_documents(dataset_path: Path, n_wallets: int, seed: int) -> list[CorpusDocument]:
    rows = select_source_rows(dataset_path, n_wallets=n_wallets, seed=seed)
    documents: list[CorpusDocument] = []

    for idx, row in rows.iterrows():
        document_id = f"derived_wallet_{idx + 1:03d}"
        text = observation_card_text(row, document_id)
        documents.append(
            make_document(
                document_id=document_id,
                document_type="observation_card",
                source_kind="derived_data",
                wallet_address=str(row["wallet_address"]),
                source_row_ids=[int(row["source_row_id"])],
                snapshot_time=timestamp_to_iso(row.get("borrow_timestamp")),
                text=text,
            )
        )

    for idx, row in rows.head(5).iterrows():
        document_id = f"synthetic_note_{idx + 1:03d}"
        text = synthetic_note_text(row, document_id)
        documents.append(
            make_document(
                document_id=document_id,
                document_type="analyst_note",
                source_kind="synthetic_demo",
                wallet_address=str(row["wallet_address"]),
                source_row_ids=[int(row["source_row_id"])],
                snapshot_time=timestamp_to_iso(row.get("borrow_timestamp")),
                text=text,
            )
        )

    for document_id, document_type, text in methodology_documents():
        source_kind = "synthetic_demo" if "synthetic_demo" in text else "derived_data"
        documents.append(
            make_document(
                document_id=document_id,
                document_type=document_type,
                source_kind=source_kind,
                wallet_address=None,
                source_row_ids=[],
                snapshot_time=None,
                text=text,
            )
        )

    return documents


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    paragraphs = []
    offset = 0
    for block in text.split("\n\n"):
        block_start = text.find(block, offset)
        paragraphs.append((block_start, block))
        offset = block_start + len(block)
    return paragraphs


def chunk_document(document: CorpusDocument, max_chars: int = 900) -> list[CorpusChunk]:
    paragraphs = split_paragraphs(document.text)
    raw_chunks: list[tuple[int, int, str]] = []
    current_parts: list[str] = []
    current_start: int | None = None

    for start, paragraph in paragraphs:
        proposed = paragraph if not current_parts else "\n\n".join(current_parts + [paragraph])
        if current_parts and len(proposed) > max_chars:
            text = "\n\n".join(current_parts)
            raw_chunks.append((current_start or 0, (current_start or 0) + len(text), text))
            current_parts = [paragraph]
            current_start = start
        else:
            if current_start is None:
                current_start = start
            current_parts.append(paragraph)

    if current_parts:
        text = "\n\n".join(current_parts)
        raw_chunks.append((current_start or 0, (current_start or 0) + len(text), text))

    chunks = []
    for idx, (start, end, text) in enumerate(raw_chunks):
        chunk_id = f"{document.document_id}::chunk_{idx + 1:03d}"
        previous_id = f"{document.document_id}::chunk_{idx:03d}" if idx > 0 else None
        next_id = (
            f"{document.document_id}::chunk_{idx + 2:03d}"
            if idx < len(raw_chunks) - 1
            else None
        )
        chunks.append(
            CorpusChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=idx,
                text=text,
                offset_start=start,
                offset_end=end,
                text_hash=sha256_text(text),
                embedding_version=EMBEDDING_VERSION,
                index_version=INDEX_VERSION,
                previous_chunk_id=previous_id,
                next_chunk_id=next_id,
            )
        )
    return chunks


def write_corpus(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    n_wallets: int = 20,
    seed: int = DEFAULT_SEED,
    rebuild: bool = False,
) -> dict[str, Any]:
    if rebuild and output_dir.exists():
        shutil.rmtree(output_dir)
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    documents = generate_documents(dataset_path, n_wallets=n_wallets, seed=seed)
    chunks: list[CorpusChunk] = []

    for document in documents:
        target = output_dir / document.path
        target.write_text(document.text, encoding="utf-8")
        chunks.extend(chunk_document(document))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")

    with (output_dir / "document_links.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["document_id", "wallet_address", "source_kind", "source_row_ids"],
        )
        writer.writeheader()
        for document in documents:
            writer.writerow(
                {
                    "document_id": document.document_id,
                    "wallet_address": document.wallet_address or "",
                    "source_kind": document.source_kind,
                    "source_row_ids": ",".join(str(x) for x in document.source_row_ids),
                }
            )

    chunk_ids_by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_doc.setdefault(chunk.document_id, []).append(chunk.chunk_id)

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "created_at": CREATED_AT,
        "seed": seed,
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_kinds": sorted({document.source_kind for document in documents}),
        "documents": [
            {
                key: value
                for key, value in asdict(document).items()
                if key != "text"
            }
            | {"chunk_ids": chunk_ids_by_doc.get(document.document_id, [])}
            for document in documents
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
