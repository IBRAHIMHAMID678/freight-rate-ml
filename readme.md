# Spotter Freight Rate ML Prediction

Machine learning solution to predict commercial freight spot rates (`posted_rate`) across US freight lanes and generate validated benchmark forecasts.

---

## 📦 Dependencies

Install all dependencies via `pip`:

```bash
pip install -r requirements.txt
```

**Required packages:**
* `catboost>=1.2.0`
* `lightgbm>=4.0.0`
* `xgboost>=2.0.0`
* `scikit-learn>=1.4.0`
* `pandas>=2.0,<3`
* `numpy>=1.26,<3`
* `matplotlib>=3.8,<4`

---

## 🚀 Run Instructions

### 1. Evaluate CatBoost (Selected Winning Model)
Runs out-of-time validation (Jan–Aug train vs. Sep–Oct validation) for the winning CatBoost model:
```bash
python src/evaluate_catboost.py
```
> **Validation Metrics:** Val MAE: **$105.96** | Val RMSE: **$632.54** | Val R²: **0.8282**

### 2. Evaluate All Models (Full Benchmark Suite)
Runs the full comparison across Ridge, LightGBM, XGBoost, Ensembles, and CatBoost:
```bash
python src/evaluate_models.py
```

### 3. Retrain & Generate Predictions
Trains CatBoost on all 48,000 historical shipments, generates `validation_predictions.csv` (12,000 loads), predicts the December benchmark route, and runs the official scorer:
```bash
python src/train_predict_submit.py
```

### 4. Run Official Scorer
Validates submission files using Spotter's official verification script:
```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```
