import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, name="LightGBM"):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    train_roc_auc = roc_auc_score(y_train, y_pred_train)
    val_roc_auc = roc_auc_score(y_val, y_pred_val)
    test_roc_auc = roc_auc_score(y_test, y_pred_test)
    print(f"Train ROC AUC: {train_roc_auc:.4f}")
    print(f"Val   ROC AUC: {val_roc_auc:.4f}")
    print(f"Test  ROC AUC: {test_roc_auc:.4f}")

    # Select the operating threshold on validation only.
    val_precision, val_recall, val_thresholds = precision_recall_curve(y_val, y_pred_val)
    val_f1_scores = 2 * (val_precision[:-1] * val_recall[:-1]) / (
        val_precision[:-1] + val_recall[:-1] + 1e-9
    )
    best_threshold_idx = int(np.argmax(val_f1_scores))
    best_threshold = float(val_thresholds[best_threshold_idx])
    print(f"\nBest threshold on validation (by F1): {best_threshold:.4f}")
    print(f"Validation F1 at threshold: {val_f1_scores[best_threshold_idx]:.4f}")

    # Evaluate test once with the fixed validation-selected threshold.
    y_test_pred_class = (y_pred_test >= best_threshold).astype(int)
    report = classification_report(y_test, y_test_pred_class)
    report_dict = classification_report(y_test, y_test_pred_class, output_dict=True)
    print("\nClassification Report (Test) with validation-selected threshold:")
    print(report)

    with open(f"results/{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(
            "Evaluation protocol: threshold selected on validation by F1; "
            "test evaluated once with that fixed threshold.\n"
        )
        f.write(f"Train ROC AUC: {train_roc_auc:.4f}\n")
        f.write(f"Val ROC AUC: {val_roc_auc:.4f}\n")
        f.write(f"Test ROC AUC: {test_roc_auc:.4f}\n")
        f.write(f"Best threshold from validation: {best_threshold:.4f}\n\n")
        f.write(report)

    metrics = {
        "threshold_source": "validation_f1",
        "best_threshold": best_threshold,
        "validation_f1_at_threshold": float(val_f1_scores[best_threshold_idx]),
        "train_roc_auc": float(train_roc_auc),
        "val_roc_auc": float(val_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_report": report_dict,
    }
    with open(f"results/{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.savefig(f"plots/{name}_roc_curve.png")
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_test)
    avg_precision = average_precision_score(y_test, y_pred_test)
    selected_idx = int(np.argmin(np.abs(thresholds - best_threshold))) if len(thresholds) else 0
    selected_recall = recall[selected_idx] if len(recall) else 0
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    plt.axvline(
        x=selected_recall,
        color="r",
        linestyle="--",
        label=f"Val threshold = {best_threshold:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"plots/{name}_pr_curve.png")
    plt.close()

    return best_threshold


def shap_analysis(model, X_test, name="LightGBM"):
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.savefig(f"plots/{name}_shap_importance.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(f"plots/{name}_shap_beeswarm.png", bbox_inches="tight")
    plt.close()


def get_risky_wallets(model, X_full, df_full, name="LightGBM"):
    probs = model.predict_proba(X_full)[:, 1]
    scored = pd.DataFrame(
        {
            "wallet_address": (
                df_full["wallet_address"].values
                if "wallet_address" in df_full.columns
                else df_full.index.astype(str)
            ),
            "probability": probs,
            "target": df_full["target"].values if "target" in df_full.columns else np.nan,
            "source_row_index": df_full.index,
        }
    )

    # Wallet score is the maximum row-level risk for that wallet.
    scored = scored.sort_values(["wallet_address", "probability"], ascending=[True, False])
    representative_rows = scored.groupby("wallet_address", as_index=False).first()
    record_counts = scored.groupby("wallet_address").size().rename("records_count").reset_index()
    wallet_targets = (
        scored.groupby("wallet_address")["target"].max().rename("wallet_target_max").reset_index()
    )

    result = (
        representative_rows.merge(record_counts, on="wallet_address", how="left")
        .sort_values("probability", ascending=False)
        .head(50)
    )

    result[["wallet_address", "probability", "records_count", "source_row_index"]].to_csv(
        f"results/{name}_top50_risky_wallets.csv",
        index=False,
    )
    result.merge(wallet_targets, on="wallet_address", how="left").to_csv(
        f"results/{name}_top50_risky_wallets_evaluation.csv",
        index=False,
    )
