"""
Train final model, generate predictions, and run scorer.
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.data_preprocessing import (
    clean_and_prepare_features,
    extract_city_coordinates_from_train,
    add_coordinates_from_lookup
)


def main() -> None:
    # 1. Load data
    train_path = ROOT / "train-test.csv"
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df):,} training rows.")

    clean_train, feature_cols, train_median_weight = clean_and_prepare_features(
        train_df, fit=True
    )
    X_train_full = clean_train[feature_cols].astype(float)
    y_train_full = np.log1p(clean_train["posted_rate"].values)

    # 2. Train CatBoost
    print("Training CatBoost model...")
    final_model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    final_model.fit(X_train_full, y_train_full)
    print("Training complete.")

    # Feature importance
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.get_feature_importance()
    }).sort_values("Importance", ascending=False)
    importances.to_csv(analysis_dir / "feature_importance.csv", index=False)

    # 3. Predict validation set
    print("Predicting validation set (12,000 loads)...")
    val_path = ROOT / "validation.csv"
    val_df = pd.read_csv(val_path)

    clean_val, _, _ = clean_and_prepare_features(
        val_df, median_weight=train_median_weight, fit=False
    )
    X_val = clean_val[feature_cols].astype(float)
    val_pred_dollars = np.round(np.expm1(final_model.predict(X_val)), 2)

    sub_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": val_pred_dollars
    })

    sub_path_root = ROOT / "validation_predictions.csv"
    sub_df.to_csv(sub_path_root, index=False)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(data_dir / "validation_predictions.csv", index=False)
    print(f"Saved: {sub_path_root.name}")

    # 4. Predict December benchmark
    print("Predicting December inputs (31 rows)...")
    dec_path = ROOT / "december-chart-inputs.csv"
    dec_df = pd.read_csv(dec_path)

    city_coords = extract_city_coordinates_from_train(train_df)
    dec_df = add_coordinates_from_lookup(dec_df, city_coords)

    clean_dec, _, _ = clean_and_prepare_features(
        dec_df, median_weight=train_median_weight, fit=False
    )
    X_dec = clean_dec[feature_cols].astype(float)
    dec_pred_dollars = np.round(np.expm1(final_model.predict(X_dec)), 2)

    dec_output = dec_df[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
    dec_output["predicted_rate"] = dec_pred_dollars

    try:
        dec_output.to_csv(dec_path, index=False)
        print(f"Saved: {dec_path.name}")
    except (PermissionError, OSError):
        pass

    dec_output.to_csv(data_dir / "december_chart_inputs.csv", index=False)
    print("Saved: data/december_chart_inputs.csv")

    # 5. Run scorer
    print("Running score.py...")
    score_script = ROOT / "score.py"
    cmd = [
        sys.executable,
        str(score_script),
        "--predictions", str(sub_path_root),
        "--december-predictions", str(dec_path if dec_path.is_file() else (data_dir / "december_chart_inputs.csv")),
        "--output-dir", str(ROOT / "scorer_results")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
