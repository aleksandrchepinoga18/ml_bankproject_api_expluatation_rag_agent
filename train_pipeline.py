import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import pandas as pd

from src.data_preparation import (
    fill_missing_with_median,
    load_and_clean_data,
    prepare_features,
    remove_high_corr_features,
    split_data,
)
from src.eda import run_eda
from src.evaluate import evaluate_model, get_risky_wallets, shap_analysis
from src.mlflow_tracking import log_model_package, result_to_dict
from src.train import train_lightgbm


def _split_summary(train, val, test):
    parts = {"train": train, "validation": val, "test": test}
    wallet_sets = {
        name: set(part["wallet_address"].dropna()) if "wallet_address" in part.columns else set()
        for name, part in parts.items()
    }
    return {
        "unit": "row: wallet risk observation; multiple rows may share one wallet_address",
        "validation_goal": "predict risk for wallets not seen during training",
        "split_strategy": "group split by wallet_address; wallet-level stratification uses max(target)",
        "rows": {name: int(len(part)) for name, part in parts.items()},
        "unique_wallets": {name: int(len(wallets)) for name, wallets in wallet_sets.items()},
        "wallet_intersections": {
            "train_validation": int(len(wallet_sets["train"] & wallet_sets["validation"])),
            "train_test": int(len(wallet_sets["train"] & wallet_sets["test"])),
            "validation_test": int(len(wallet_sets["validation"] & wallet_sets["test"])),
        },
    }


if __name__ == "__main__":
    df = load_and_clean_data("data/dataset.parquet")
    df = remove_high_corr_features(df)

    run_eda(df)

    train, val, test = split_data(df, random_state=42, group_col="wallet_address")
    feature_cols = prepare_features(df)

    X_train_raw = train[feature_cols].copy()
    y_train = train["target"].copy()
    X_val_raw = val[feature_cols].copy()
    y_val = val["target"].copy()
    X_test_raw = test[feature_cols].copy()
    y_test = test["target"].copy()

    X_train, X_val, X_test, imputation_values = fill_missing_with_median(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        return_medians=True,
    )

    print("Remaining features:", len(X_train.columns))
    print(X_train.columns.tolist())
    print("Split complete:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    split_summary = _split_summary(train, val, test)
    print("Wallet split summary:", split_summary)

    model, best_params, feature_names_from_model = train_lightgbm(
        X_train,
        y_train,
        X_val[feature_cols],
        y_val,
    )
    joblib.dump(feature_names_from_model, "models/lightgbm_feature_names.pkl")
    joblib.dump(best_params, "models/lightgbm_best_params.pkl")
    joblib.dump(
        {
            "feature_names": feature_names_from_model,
            "imputation_values": imputation_values,
            "preprocessing": "replace inf with nan; fill missing values with train medians",
            "split_summary": split_summary,
        },
        "models/lightgbm_preprocessing.pkl",
    )

    best_threshold = evaluate_model(
        model,
        X_train,
        X_val[feature_names_from_model],
        X_test[feature_names_from_model],
        y_train,
        y_val,
        y_test,
        name="LightGBM",
    )

    shap_analysis(model, X_test[feature_names_from_model], name="LightGBM")
    X_full = pd.concat([X_train, X_val, X_test])[feature_names_from_model]
    df_full = pd.concat([train, val, test]).reset_index(drop=True)
    get_risky_wallets(model, X_full, df_full, name="LightGBM")

    print("Saving reference data for monitoring...")
    os.makedirs("monitoring/reference", exist_ok=True)

    X_train_for_ref = X_train[feature_names_from_model].copy()
    train_scores = model.predict_proba(X_train_for_ref)[:, 1]

    X_train_for_ref.to_parquet("monitoring/reference/reference_features.parquet", index=False)
    pd.DataFrame({"score": train_scores}).to_parquet(
        "monitoring/reference/reference_scores.parquet",
        index=False,
    )

    X_train_for_ref.to_csv("monitoring/reference/reference_features.csv", index=False)
    pd.DataFrame({"score": train_scores}).to_csv(
        "monitoring/reference/reference_scores.csv",
        index=False,
    )

    joblib.dump(best_threshold, "models/lightgbm_best_threshold.pkl")
    print("Reference data saved in monitoring/reference/ (parquet + csv)")
    print(f"Best threshold saved: {best_threshold:.4f}")
    print(f"Features used: {len(feature_names_from_model)}")
    print("Training complete!")

    if os.getenv("MLFLOW_TRACKING_DISABLED") != "1":
        mlflow_result = log_model_package(run_name="train_pipeline")
        print("MLflow logging result:", result_to_dict(mlflow_result))

    print("Example API input:")
    print(X_test[feature_names_from_model].iloc[0].to_dict())
