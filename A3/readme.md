# Smart Factory Alert Agent (a3.ipynb)

## Overview
This notebook simulates a smart-factory equipment anomaly detection pipeline that combines strict rule-based checks with a supervised ML model (RandomForest) and SHAP explanations. It generates synthetic sensor data, injects anomalies and missing values, trains a RandomForest on an initial window, explains ML anomalies with SHAP, computes rule scores, merges alerts, and visualizes results.

## Dependencies
- pandas, numpy
- scikit-learn (StandardScaler, RandomForestClassifier)
- shap (TreeExplainer)
- matplotlib
Note: the notebook contains `%pip install matplotlib` and `%pip install shap` cells.

## Data generation
Function: `generate_dummy_data`(
- n_rows=300,
- start_time="2025-01-07 1:00:00",
- interval_minutes=1,
- anomaly_rate=0.15,
- introduce_missing=False,
- missing_rate=0.01
)

Columns per row: timestamp, temp, pressure, vibration, label.

Normal ranges (used when not injecting anomaly):
- temp: 45.0 – 50.0 (°C)
- pressure: 1.00 – 1.05 (bar)
- vibration: 0.02 – 0.04

Injected anomaly sampling (when chosen):
- temp anomaly: either <4 3 or > 52
- pressure anomaly: either <0.97 or >1.08
- vibration anomaly: >0.07

Labeling: label = "abnormal" if any sensor is outside its strict threshold:
- temp outside 43–52
- pressure outside 0.97–1.08
- vibration > 0.07

Missing values: when `introduce_missing=True`, random sensor cells are set to `NaN` based on `missing_rate`.

Example run in notebook: g`enerate_dummy_data(n_rows=150, interval_minutes=5, anomaly_rate=0.15, introduce_missing=True)`

## Preprocessing
- Missing counts are shown.
- A forward-fill call is executed via `df.ffill()`
- Scaling will be performed in the next step

## Machine learning (supervised)
- Features: ["temp", "pressure", "vibration"] are scaled with StandardScaler.
- Model: RandomForestClassifier(n_estimators=100, random_state=50, class_weight="balanced")
- Training: model.fit on the first 50 rows (X_train = scaled first 50 rows, y_train = corresponding labels mapped {normal:0, abnormal:1})
- Prediction: predictions and predict_proba are run on all rows; results are stored in `ml_pred` and `ml_score`.
- ML anomalies: rows where `ml_pred == 1` are collected as `ml_anomalies`.

## SHAP explanations
- Uses `shap.TreeExplainer(clf, data=X_train, model_output="probability")`.
- Explanations computed only for rows flagged as ML anomalies.
- For each anomalous row, features with positive SHAP contribution `> 0.1` are reported.
- Explanations are added as `ml_explanation` to `ml_anomalies`.

## Rule-based scoring & detection
- detect_anomalies(row): returns human-readable reasons:
  - "Temperature out of range (X°C)"
  - "Pressure out of range (X bar)"
  - "High vibration (X)"
- anomaly_score(temp, pressure, vibration):
  - temp contribution: abs(distance beyond 43–52) added directly
  - pressure contribution: abs(distance beyond 0.97–1.08) * 10
  - vibration contribution: (vibration - 0.07) * 100 if > 0.07
  - result rounded to 4 decimals
- rule_anomalies DataFrame contains timestamp, temp, pressure, vibration, score, alert_reasons.

## Combining alerts
Function: `build_combined_alerts(df_rule, df_ml)`
- Ensures timestamps are datetime.
- Computes rule_score = anomaly_score(...) per rule alert, then min-max normalizes rule_score across rule alerts.
- Prepares flags: `rule=True` for rule alerts, `ml=True` for ml alerts.
- Outer-joins (merge) rule and ML alerts on timestamp.

Display: `combined_alerts(merged_alerts)` prints a formatted alert list showing:
- detection source (Rule-based, ML-based, or Both)
- sensor values
- Rule Score and Anamoly reasons
- ML Anomaly Score and ML Suggestion 

## Visualization
Three matplotlib plots (temperature, pressure, vibration) with anomalies highlighted
