# Wallet aggregation analysis

Дата: 2026-09-29.

Цель: проверить базовое правило `max_row_probability`, влияние `records_count` и сравнить его с заранее заданной альтернативой `mean_row_probability`.

## Method

- Использован локальный `data/dataset.parquet` и сохраненный LightGBM artifact.
- Split и preprocessing воспроизведены тем же кодом, что и train pipeline.
- Базовое правило не меняется: `max_row_probability` остается основным.
- Альтернатива для сравнения: `mean_row_probability`.

## Results

- Wallets: `74476`.
- Rows: `442961`.
- records_count min/median/p95/max: `1` / `1.0` / `21.0` / `2759`.
- Spearman records_count vs max_row_probability: `0.2901352795602595`.
- Spearman records_count vs max_rank: `-0.29013527954119495`.
- Top-50 overlap max vs mean: `4/50` (`0.08`).
- Baseline-only top-50 wallets: `46`.
- Alternative-only top-50 wallets: `46`.
- Baseline top-50 records_count median: `214.5`.
- Alternative top-50 records_count median: `2.0`.

## Limitations

- Это offline-анализ на доступных данных, не production-мониторинг.
- Он не доказывает качество wallet-level решения.
- Temporal leakage review из `docs/model_boundary.md` остается открытым.
- Базовое правило не изменено без отдельного решения.
