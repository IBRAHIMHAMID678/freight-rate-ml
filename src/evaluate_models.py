"""
Temporal validation benchmark comparing Ridge, CatBoost, LightGBM, and XGBoost.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from src.data_preprocessing import clean_and_prepare_features


def run_temporal_validation() -> pd.DataFrame:
    raw_df = pd.read_csv(ROOT / "train-test.csv")
    raw_df["date_dt"] = pd.to_datetime(raw_df["date"])

    # Temporal holdout: Jan-Aug train, Sep-Oct validation
    tr_mask = raw_df["date_dt"] < "2025-09-01"
    va_mask = raw_df["date_dt"] >= "2025-09-01"

    print(f"Data split: {tr_mask.sum():,} train (Jan-Aug), {va_mask.sum():,} val (Sep-Oct)")

    # Preprocessing fitted strictly on train split
    train_median_weight = float(raw_df.loc[tr_mask, "weight"].abs().median())
    df, feature_cols, _ = clean_and_prepare_features(
        raw_df, median_weight=train_median_weight, fit=False
    )

    X_train = df.loc[tr_mask, feature_cols].astype(float)
    y_train = np.log1p(df.loc[tr_mask, "posted_rate"].values)

    X_val = df.loc[va_mask, feature_cols].astype(float)
    y_val_dollars = df.loc[va_mask, "posted_rate"].values

    results = []

    # 1. Ridge baseline
    print("Evaluating Ridge baseline...")
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0))
    ])
    ridge_pipe.fit(X_train, y_train)
    p_ridge = np.expm1(ridge_pipe.predict(X_val))
    results.append({
        "Model": "1. Baseline (Ridge)",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_ridge),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_ridge),
        "Val R2": r2_score(y_val_dollars, p_ridge),
    })

    # 2. CatBoost
    print("Evaluating CatBoost...")
    cb = CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cb.fit(X_train, y_train)
    p_cb = np.expm1(cb.predict(X_val))
    results.append({
        "Model": "2. CatBoost",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_cb),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_cb),
        "Val R2": r2_score(y_val_dollars, p_cb),
    })

    # 3. LightGBM
    print("Evaluating LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    p_lgb = np.expm1(lgb_model.predict(X_val))
    results.append({
        "Model": "3. LightGBM",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_lgb),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_lgb),
        "Val R2": r2_score(y_val_dollars, p_lgb),
    })

    # 4. XGBoost
    print("Evaluating XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=700,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = np.expm1(xgb_model.predict(X_val))
    results.append({
        "Model": "4. XGBoost",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_xgb),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_xgb),
        "Val R2": r2_score(y_val_dollars, p_xgb),
    })

    # 5. Ensembles
    print("Evaluating Ensembles...")
    p_ens_tri = (p_cb + p_lgb + p_xgb) / 3.0
    results.append({
        "Model": "5. Ensemble (CB + LGB + XGB)",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_ens_tri),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_ens_tri),
        "Val R2": r2_score(y_val_dollars, p_ens_tri),
    })

    p_ens_dual = 0.5 * p_lgb + 0.5 * p_cb
    results.append({
        "Model": "6. Ensemble (LGB + CB 50/50)",
        "Val MAE ($)": mean_absolute_error(y_val_dollars, p_ens_dual),
        "Val RMSE ($)": root_mean_squared_error(y_val_dollars, p_ens_dual),
        "Val R2": r2_score(y_val_dollars, p_ens_dual),
    })

    # Results table
    summary_df = pd.DataFrame([{
        "Model": r["Model"],
        "Val MAE ($)": f"${r['Val MAE ($)']:.2f}",
        "Val RMSE ($)": f"${r['Val RMSE ($)']:.2f}",
        "Val R2": f"{r['Val R2']:.4f}",
    } for r in results])

    print("\nValidation Results:")
    print(summary_df.to_string(index=False))

    # Save to CSV
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(analysis_dir / "model_comparison.csv", index=False)

    return summary_df


if __name__ == "__main__":
    run_temporal_validation()
