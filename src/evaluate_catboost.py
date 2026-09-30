

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from catboost import CatBoostRegressor

from src.data_preprocessing import clean_and_prepare_features


def get_train_data_path() -> Path:
    p_root = ROOT / "train-test.csv"
    if p_root.is_file():
        return p_root
    p_data = ROOT / "data" / "train_test.csv"
    if p_data.is_file():
        return p_data
    raise FileNotFoundError(
        "Could not find training data. Please place 'train-test.csv' in the repository root or data/."
    )


def evaluate_catboost():
    data_path = get_train_data_path()
    print(f"Loading data from: {data_path.name}...")
    raw_df = pd.read_csv(data_path)
    raw_df["date_dt"] = pd.to_datetime(raw_df["date"])

    # Temporal holdout: Jan-Aug train, Sep-Oct validation (mirrors 2-month test horizon)
    tr_mask = raw_df["date_dt"] < "2025-09-01"
    va_mask = raw_df["date_dt"] >= "2025-09-01"

    print(f"Split: {tr_mask.sum():,} train loads (Jan-Aug) | {va_mask.sum():,} validation loads (Sep-Oct)")

    # Data hygiene fitted strictly on training partition
    train_median_weight = float(raw_df.loc[tr_mask, "weight"].abs().median())
    df, feature_cols, _ = clean_and_prepare_features(
        raw_df, median_weight=train_median_weight, fit=False
    )

    X_train = df.loc[tr_mask, feature_cols].astype(float)
    y_train = np.log1p(df.loc[tr_mask, "posted_rate"].values)

    X_val = df.loc[va_mask, feature_cols].astype(float)
    y_val_dollars = df.loc[va_mask, "posted_rate"].values

    print("\nTraining winning CatBoost model (700 trees, depth 6, lr=0.05)...")
    cb = CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cb.fit(X_train, y_train)

    # Predict in dollars
    p_cb = np.expm1(cb.predict(X_val))

    mae = mean_absolute_error(y_val_dollars, p_cb)
    rmse = root_mean_squared_error(y_val_dollars, p_cb)
    r2 = r2_score(y_val_dollars, p_cb)

    print("\n" + "=" * 50)
    print("       CatBoost Out-of-Time Validation Results")
    print("=" * 50)
    print(f" Validation MAE  : ${mae:.2f}")
    print(f" Validation RMSE : ${rmse:.2f}")
    print(f" Validation R2   : {r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_catboost()
