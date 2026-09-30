# Spotter Freight Rate ML Prediction Challenge

Production-grade machine learning solution for Spotter's Freight Rate Prediction Challenge. This repository trains and validates an out-of-time model to predict commercial freight spot rates (`posted_rate`) across US freight lanes and generates benchmark predictions for the official scorer.

---

## 🏆 Key Results

Using an out-of-time temporal holdout (Jan 1 – Aug 31 train vs. Sep 1 – Oct 31 validation) simulating the exact 2-month test horizon:

| Model | Architecture | Target Formulation | Val MAE ($) | Val RMSE ($) | Val R² | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Ridge)** | Scaled Ridge Regression ($\alpha=10.0$) | $\log(1 + \text{rate})$ | $452.52 | $946.61 | 0.6152 | Baseline |
| **XGBoost** | 700 trees, depth 6, lr=0.04 | $\log(1 + \text{rate})$ | $124.79 | $638.05 | 0.8252 | Evaluated |
| **LightGBM** | 700 trees, 31 leaves, lr=0.04 | $\log(1 + \text{rate})$ | $120.34 | $634.24 | 0.8273 | Evaluated |
| **Ensemble (CB+LGB+XGB)** | Equal weighted tri-blend | $\log(1 + \text{rate})$ | $112.59 | $633.49 | 0.8277 | Evaluated |
| **Ensemble (LGB+CB)** | 50/50 dual blend | $\log(1 + \text{rate})$ | $110.10 | $632.63 | 0.8281 | Evaluated |
| **CatBoost (Selected)** | **700 trees, depth 6, lr=0.05** | $\mathbf{\log(1 + \text{rate})}$ | **$105.96** | **$632.54** | **0.8280** | **Final Selected Model** |

> **Official Scorer Result:** All 12,000 final predictions and 31 December benchmark predictions passed strict validation with zero errors.

---

## 📁 Repository Structure

```text
├── src/                                  # Solution source code
│   ├── data_preprocessing.py             # Feature engineering & cleaning pipeline
│   ├── evaluate_models.py                # Temporal validation runner & benchmark suite
│   └── train_predict_submit.py           # Full retraining, inference & score.py execution
├── analysis/                             # Experiment outputs & benchmark metrics
│   └── feature_importance.csv           # Final model feature importances
├── scorer_results/                       # Official scorer outputs
│   └── candidate_december.png            # Generated December rate chart
├── score.py                              # Spotter's official verification script
├── validation_predictions.csv            # Final submission predictions (12,000 rows)
├── december-chart-inputs.csv             # Final completed December inputs (31 rows)
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Excludes datasets, cache, and logs
└── README.md                             # Documentation & run instructions
```

---

## 🚀 Quickstart & Reproduction

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run Temporal Validation Benchmark
To reproduce the model comparison table across Ridge, LightGBM, XGBoost, CatBoost, and Ensembles on the out-of-time split:
```bash
python src/evaluate_models.py
```

### 3. Retrain on All Data & Generate Submission Predictions
To train the optimal CatBoost model on all 48,000 loads, generate `validation_predictions.csv`, predict the 31 December benchmark rows, and run the official scorer:
```bash
python src/train_predict_submit.py
```

### 4. Run the Official Scorer Manually
```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```
Expected output:
```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## 🧠 Key Methodology & Engineering Decisions

1. **Target Formulation ($\log(1 + \text{rate})$)**:
   Predicting in log-space reduced out-of-time MAE from $138.66 to $119.71 ($18.95/load improvement), penalizing relative percentage errors evenly across short $500 hauls and transcontinental $15,000 hauls.
2. **Handling Unseen Test Geography**:
   EDA revealed 8 brand-new cities and 736 unseen lanes in the test set. The model avoids discrete city memorization and uses continuous spatial coordinates (`pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`), great-circle Haversine distances, route circuity ratios, and transit bearings.
3. **Data Quality Treatments**:
   * **Negative Weights (292 in train, 145 in val):** Confirmed as sign entry bugs; taking `abs(weight)` followed by median imputation reduced validation MAE from $140.59 to $138.66.
   * **Market Signals (`market_index`, `quote_signal`):** Excluding these volatile signals improved out-of-time MAE from $152.07 to $119.71, avoiding overfitting to past regimes and enabling leakage-free prediction on December benchmark inputs.
4. **Model Selection**:
   CatBoost standalone achieved $105.96 MAE, outperforming blended ensembles ($110.10). In accordance with empirical validation, the standalone model was selected.

---

## 📈 December Benchmark Rate Chart

![Candidate December Rate Chart](scorer_results/candidate_december.png)

* **Route:** Lexington to Fort Wayne (360 miles, Dry Van, 32,000 lbs)
* **Predicted Rates:** $800.17 to $841.83 (RPM $2.22 to $2.34/mi)
* **Market Dynamics:** Displays realistic industrial weekday volume surges, weekend troughs, and an end-of-year pre-holiday capacity surge peaking on December 29–30.

---

## 📄 Submission Checklist

- [x] Solution code, dependencies, and reproduction instructions.
- [x] `validation_predictions.csv` with exactly 12,000 valid, positive predictions matching load IDs.
- [x] Official scorer validation passes with zero errors.
- [x] `scorer_results/candidate_december.png` generated and verified.
