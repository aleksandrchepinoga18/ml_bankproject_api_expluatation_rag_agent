import os

import joblib
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler


BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_PARAMS = {
    "colsample_bytree": 0.70,
    "learning_rate": 0.035,
    "max_depth": 6,
    "min_child_samples": 120,
    "min_split_gain": 0.05,
    "n_estimators": 1000,
    "num_leaves": 31,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "subsample": 0.85,
}

PARAM_DISTRIBUTIONS = {
    "colsample_bytree": uniform(0.55, 0.30),
    "learning_rate": uniform(0.02, 0.04),
    "max_depth": randint(4, 8),
    "min_child_samples": randint(80, 201),
    "min_split_gain": uniform(0.0, 0.15),
    "n_estimators": randint(700, 1401),
    "num_leaves": randint(15, 64),
    "reg_alpha": uniform(0.5, 2.5),
    "reg_lambda": uniform(1.0, 4.0),
    "subsample": uniform(0.70, 0.20),
}


def _fit_candidate(params, X_train, y_train, X_val=None, y_val=None, random_state=42):
    final_params = {**BASE_PARAMS, **params, "random_state": random_state}
    model = lgb.LGBMClassifier(**final_params)
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, random_state=42, n_iter=8):
    """Train LightGBM and select hyperparameters on validation only."""
    os.makedirs("models", exist_ok=True)

    candidates = [DEFAULT_PARAMS]
    if X_val is not None and y_val is not None and n_iter > 0:
        sampled = list(
            ParameterSampler(
                PARAM_DISTRIBUTIONS,
                n_iter=n_iter,
                random_state=random_state,
            )
        )
        candidates.extend(sampled)

    best_model = None
    best_params = None
    best_val_auc = None
    tuning_results = []

    for idx, params in enumerate(candidates, 1):
        model = _fit_candidate(params, X_train, y_train, X_val, y_val, random_state)
        if X_val is not None and y_val is not None:
            val_score = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_score)
        else:
            val_auc = None

        tuning_results.append(
            {
                "candidate": idx,
                "params": dict(params),
                "val_roc_auc": None if val_auc is None else float(val_auc),
            }
        )
        print(f"LightGBM candidate {idx}/{len(candidates)} val ROC-AUC: {val_auc}")

        if best_model is None or (val_auc is not None and val_auc > best_val_auc):
            best_model = model
            best_params = dict(params)
            best_val_auc = val_auc

    joblib.dump(best_model, "models/lightgbm_model.pkl")
    joblib.dump(
        {
            "selection_metric": "validation_roc_auc",
            "best_val_roc_auc": None if best_val_auc is None else float(best_val_auc),
            "best_params": best_params,
            "candidates": tuning_results,
        },
        "models/lightgbm_param_search.pkl",
    )

    return best_model, best_params, X_train.columns.tolist()
