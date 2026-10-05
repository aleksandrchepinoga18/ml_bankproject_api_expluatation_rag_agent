# Demo document corpus

Дата: 2026-09-29.

Документ фиксирует результат команды 6 из
`TZ_Scoring_RAG_Agent_Production_v2.md`.

## Scope

Корпус является демонстрационным и воспроизводимым. Он не содержит реальных
внешних расследований, жалоб, юридических заключений или подтвержденных фактов
мошенничества.

## Outputs

Корпус собирается командой:

```bash
python scripts/build_demo_corpus.py --rebuild
```

Файлы:

- `data/corpus/documents/*.md` - исходные тексты для аудита;
- `data/corpus/manifest.json` - manifest документов и связей;
- `data/corpus/chunks.jsonl` - результаты chunking;
- `data/corpus/document_links.csv` - плоская таблица связей документов.

## Source Kinds

- `derived_data`: карточки наблюдений и общие методические документы. Карточки
  строятся только из реально существующих полей `data/dataset.parquet`.
- `synthetic_demo`: явно маркированные демонстрационные заметки и нерелевантный
  текст для проверки поиска.

## Boundary

Карточки наблюдений не являются независимым доказательством и не превращают
модельный score в факт мошенничества. В них не записывается score LightGBM.

Для address-specific retrieval точная связь `wallet_address` должна проверяться
до семантического поиска. Общие методические документы не утверждают фактов о
конкретном кошельке.

## Chunking

Chunking сохраняет границы абзацев и заголовков. Каждый chunk содержит:

- `chunk_id`;
- `document_id`;
- `offset_start` / `offset_end`;
- `text_hash`;
- `embedding_version`;
- `index_version`;
- `previous_chunk_id` / `next_chunk_id`.

`embedding_version` и `index_version` пока имеют значение `not_indexed_v1`,
потому что Qdrant/embeddings относятся к следующей команде.
