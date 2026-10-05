import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(path: str):
    df = pd.read_parquet(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df

def remove_high_corr_features(df: pd.DataFrame):
    removed_features = [
        "market_rocp", "market_apo", "market_macdsignal_macdfix",
        "market_macd_macdfix", "borrow_block_number", "borrow_timestamp",
        "risky_first_tx_timestamp", "market_macd_macdext", "market_macd",
        "market_macdsignal", "liquidation_count"
    ]
    to_remove = [f for f in removed_features if f in df.columns and f != 'wallet_address']
    df = df.drop(columns=to_remove, errors="ignore")
    return df

def split_data(df: pd.DataFrame, random_state: int = 42, group_col: str = "wallet_address"):
    if group_col not in df.columns:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train, temp = train_test_split(
            df,
            test_size=0.5,
            random_state=random_state,
            stratify=df["target"] if "target" in df.columns else None,
        )
        val, test = train_test_split(
            temp,
            test_size=0.5,
            random_state=random_state,
            stratify=temp["target"] if "target" in temp.columns else None,
        )
        return train, val, test

    wallet_labels = (
        df.groupby(group_col, dropna=False)["target"]
        .max()
        .rename("wallet_target")
        .reset_index()
    )
    train_wallets, temp_wallets = train_test_split(
        wallet_labels,
        test_size=0.5,
        random_state=random_state,
        stratify=wallet_labels["wallet_target"],
    )
    val_wallets, test_wallets = train_test_split(
        temp_wallets,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_wallets["wallet_target"],
    )

    train = df[df[group_col].isin(train_wallets[group_col])].copy()
    val = df[df[group_col].isin(val_wallets[group_col])].copy()
    test = df[df[group_col].isin(test_wallets[group_col])].copy()
    return train, val, test

def prepare_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['target', 'wallet_address']
    return [col for col in df.columns if col not in exclude_cols]

def fit_missing_medians(X_train: pd.DataFrame) -> dict:
    medians = {}
    for col in X_train.columns:
        if X_train[col].isna().any():
            med = X_train[col].median()
            medians[col] = 0.0 if pd.isna(med) else float(med)
    return medians

def apply_missing_medians(X: pd.DataFrame, medians: dict) -> pd.DataFrame:
    X = X.copy()
    for col, med in medians.items():
        if col in X.columns:
            X[col] = X[col].fillna(med)
    return X

def fill_missing_with_median(X_train, X_val, X_test, return_medians=False):
    medians = fit_missing_medians(X_train)
    X_train = apply_missing_medians(X_train, medians)
    X_val = apply_missing_medians(X_val, medians)
    X_test = apply_missing_medians(X_test, medians)
    if return_medians:
        return X_train, X_val, X_test, medians
    return X_train, X_val, X_test
