import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (  # noqa: E402
    fill_missing_with_median,
    load_and_clean_data,
    prepare_features,
    remove_high_corr_features,
    split_data,
)


DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
REPORT_PATH = PROJECT_ROOT / "docs" / "aggregation_analysis.md"
RESULT_PATH = PROJECT_ROOT / "results" / "wallet_aggregation_comparison.json"


def _rank_frame(frame, score_column, rank_column):
    ranked = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    ranked[rank_column] = ranked.index + 1
    return ranked


def build_scored_rows():
    df = remove_high_corr_features(load_and_clean_data(str(DATASET_PATH)))
    train, validation, test = split_data(df, random_state=42, group_col="wallet_address")
    feature_cols = prepare_features(df)

    model = joblib.load(PROJECT_ROOT / "models" / "lightgbm_model.pkl")
    feature_names = list(joblib.load(PROJECT_ROOT / "models" / "lightgbm_feature_names.pkl"))

    X_train_raw = train[feature_cols].copy()
    X_val_raw = validation[feature_cols].copy()
    X_test_raw = test[feature_cols].copy()
    X_train, X_val, X_test = fill_missing_with_median(X_train_raw, X_val_raw, X_test_raw)

    X_full = pd.concat([X_train, X_val, X_test], ignore_index=True)[feature_names]
    df_full = pd.concat([train, validation, test], ignore_index=True)
    probabilities = model.predict_proba(X_full)[:, 1]

    return pd.DataFrame(
        {
            "wallet_address": df_full["wallet_address"].astype(str),
            "row_probability": probabilities,
            "source_row_index": df_full.index,
        }
    )


def analyze():
    scored = build_scored_rows()
    grouped = scored.groupby("wallet_address")
    wallet_scores = grouped.agg(
        max_row_probability=("row_probability", "max"),
        mean_row_probability=("row_probability", "mean"),
        records_count=("row_probability", "size"),
    ).reset_index()

    representative_rows = (
        scored.sort_values(["wallet_address", "row_probability"], ascending=[True, False])
        .groupby("wallet_address", as_index=False)
        .first()[["wallet_address", "source_row_index"]]
        .rename(columns={"source_row_index": "representative_row_id"})
    )
    wallet_scores = wallet_scores.merge(representative_rows, on="wallet_address", how="left")
    wallet_scores = _rank_frame(wallet_scores, "max_row_probability", "max_rank")
    wallet_scores = _rank_frame(wallet_scores, "mean_row_probability", "mean_rank")

    max_top50 = set(
        wallet_scores.nsmallest(50, "max_rank")["wallet_address"].astype(str)
    )
    mean_top50 = set(
        wallet_scores.nsmallest(50, "mean_rank")["wallet_address"].astype(str)
    )
    top50_overlap = len(max_top50 & mean_top50)

    result = {
        "method": {
            "baseline_rule": "max_row_probability",
            "alternative_rule": "mean_row_probability",
            "dataset": "data/dataset.parquet",
            "notes": [
                "Offline comparison uses the available local dataset and saved LightGBM artifact.",
                "The baseline rule is not changed by this analysis.",
            ],
        },
        "wallets": int(len(wallet_scores)),
        "rows": int(len(scored)),
        "records_count": {
            "min": int(wallet_scores["records_count"].min()),
            "median": float(wallet_scores["records_count"].median()),
            "p95": float(wallet_scores["records_count"].quantile(0.95)),
            "max": int(wallet_scores["records_count"].max()),
        },
        "correlations": {
            "spearman_records_count_vs_max_probability": float(
                wallet_scores["records_count"].corr(
                    wallet_scores["max_row_probability"],
                    method="spearman",
                )
            ),
            "spearman_records_count_vs_max_rank": float(
                wallet_scores["records_count"].corr(wallet_scores["max_rank"], method="spearman")
            ),
        },
        "top50_comparison": {
            "baseline_top50_rule": "max_row_probability",
            "alternative_top50_rule": "mean_row_probability",
            "overlap_count": int(top50_overlap),
            "overlap_share": float(top50_overlap / 50),
            "baseline_only_count": int(len(max_top50 - mean_top50)),
            "alternative_only_count": int(len(mean_top50 - max_top50)),
            "baseline_top50_records_count_median": float(
                wallet_scores[wallet_scores["wallet_address"].isin(max_top50)][
                    "records_count"
                ].median()
            ),
            "alternative_top50_records_count_median": float(
                wallet_scores[wallet_scores["wallet_address"].isin(mean_top50)][
                    "records_count"
                ].median()
            ),
        },
    }
    return result


def write_report(result):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = [
        "# Wallet aggregation analysis",
        "",
        "Дата: 2026-09-29.",
        "",
        "Цель: проверить базовое правило `max_row_probability`, влияние `records_count` "
        "и сравнить его с заранее заданной альтернативой `mean_row_probability`.",
        "",
        "## Method",
        "",
        "- Использован локальный `data/dataset.parquet` и сохраненный LightGBM artifact.",
        "- Split и preprocessing воспроизведены тем же кодом, что и train pipeline.",
        "- Базовое правило не меняется: `max_row_probability` остается основным.",
        "- Альтернатива для сравнения: `mean_row_probability`.",
        "",
        "## Results",
        "",
        f"- Wallets: `{result['wallets']}`.",
        f"- Rows: `{result['rows']}`.",
        f"- records_count min/median/p95/max: "
        f"`{result['records_count']['min']}` / `{result['records_count']['median']}` / "
        f"`{result['records_count']['p95']}` / `{result['records_count']['max']}`.",
        f"- Spearman records_count vs max_row_probability: "
        f"`{result['correlations']['spearman_records_count_vs_max_probability']}`.",
        f"- Spearman records_count vs max_rank: "
        f"`{result['correlations']['spearman_records_count_vs_max_rank']}`.",
        f"- Top-50 overlap max vs mean: "
        f"`{result['top50_comparison']['overlap_count']}/50` "
        f"(`{result['top50_comparison']['overlap_share']}`).",
        f"- Baseline-only top-50 wallets: "
        f"`{result['top50_comparison']['baseline_only_count']}`.",
        f"- Alternative-only top-50 wallets: "
        f"`{result['top50_comparison']['alternative_only_count']}`.",
        f"- Baseline top-50 records_count median: "
        f"`{result['top50_comparison']['baseline_top50_records_count_median']}`.",
        f"- Alternative top-50 records_count median: "
        f"`{result['top50_comparison']['alternative_top50_records_count_median']}`.",
        "",
        "## Limitations",
        "",
        "- Это offline-анализ на доступных данных, не production-мониторинг.",
        "- Он не доказывает качество wallet-level решения.",
        "- Temporal leakage review из `docs/model_boundary.md` остается открытым.",
        "- Базовое правило не изменено без отдельного решения.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result = analyze()
    write_report(result)
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
