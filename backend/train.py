import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

DATA_PATH         = "model/gas_fee_historical.csv"
MODEL_DIR         = "model"
FEE_MODEL_LOWER   = f"{MODEL_DIR}/lgbm_fee_lower.txt"
FEE_MODEL_MEDIAN  = f"{MODEL_DIR}/lgbm_fee_median.txt"
FEE_MODEL_UPPER   = f"{MODEL_DIR}/lgbm_fee_upper.txt"
RATIO_MODEL_PATH  = f"{MODEL_DIR}/lgbm_ratio_median.txt"
REPORT_PATH       = f"{MODEL_DIR}/training_report.json"

TEST_HOURS        = 24
MIN_TRAIN_DAYS    = 3
MAPE_THRESHOLD    = 25.0

BASE_L2_FLOOR_GWEI = 0.005
FLOOR_TOLERANCE    = 0.0001
SPIKE_Z_THRESHOLD  = 2.0
ROLLING_WINDOW_HOURS = 6

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "verbose": -1,
    "n_estimators": 250
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)
    if mask.sum() < 5:
        mae = float(np.mean(np.abs(actual - predicted)))
        mean_val = float(np.mean(actual))
        return (mae / mean_val * 100) if mean_val > 0 else None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, is_spike_actual: np.ndarray = None) -> dict:
    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = calculate_mape(actual, predicted)
    within_10pct = float(np.mean(np.abs(actual - predicted) / (actual + 1e-10) < 0.10) * 100)
    
    spike_metrics = {}
    if is_spike_actual is not None and len(is_spike_actual) == len(actual):
        is_spike_pred = predicted > (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)
        precision = float(precision_score(is_spike_actual, is_spike_pred, zero_division=0))
        recall = float(recall_score(is_spike_actual, is_spike_pred, zero_division=0))
        f1 = float(f1_score(is_spike_actual, is_spike_pred, zero_division=0))
        
        spike_mask = is_spike_actual
        if spike_mask.sum() > 0:
            spike_mae = float(mean_absolute_error(actual[spike_mask], predicted[spike_mask]))
        else:
            spike_mae = None
            
        spike_metrics = {"precision": precision, "recall": recall, "f1": f1, "spike_mae": spike_mae}
        
    return {"mae": mae, "rmse": rmse, "mape": mape, "within_10pct": within_10pct, **spike_metrics}

log(f"Loading data from {DATA_PATH}...")
assert os.path.exists(DATA_PATH), f"File tidak ditemukan: {DATA_PATH}"

df_raw = pd.read_csv(DATA_PATH)
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True).dt.tz_localize(None)
df_raw = df_raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
df_raw = df_raw[(df_raw["base_fee_gwei"] > 0) & (df_raw["base_fee_gwei"] < 1000)]

has_ratio = "gas_used_ratio" in df_raw.columns

log("Feature engineering: Mencegah Data Leakage (T-1) & Ekstraksi Fitur...")

df_raw["hour"] = df_raw["timestamp"].dt.hour
df_raw["day_of_week"] = df_raw["timestamp"].dt.dayofweek

ROLLING_W = int(ROLLING_WINDOW_HOURS * 12)

shifted_fee = df_raw["base_fee_gwei"].shift(1)

df_raw["fee_lag_1"] = shifted_fee
df_raw["fee_rolling_mean"] = shifted_fee.rolling(ROLLING_W, min_periods=6).mean()
df_raw["fee_rolling_std"]  = shifted_fee.rolling(ROLLING_W, min_periods=6).std()
df_raw["fee_rolling_p75"]  = shifted_fee.rolling(ROLLING_W, min_periods=6).quantile(0.75)

if has_ratio:
    df_raw["gas_used_ratio"] = df_raw["gas_used_ratio"].fillna(df_raw["gas_used_ratio"].median())
    shifted_ratio = df_raw["gas_used_ratio"].shift(1)
    
    df_raw["ratio_lag_1"] = shifted_ratio
    df_raw["ratio_rolling_mean"] = shifted_ratio.rolling(ROLLING_W, min_periods=6).mean()
    df_raw["ratio_rolling_std"]  = shifted_ratio.rolling(ROLLING_W, min_periods=6).std()

df_raw = df_raw.dropna().reset_index(drop=True)

rolling_mean_eval = df_raw["base_fee_gwei"].rolling(window=288, min_periods=12).mean()
rolling_std_eval  = df_raw["base_fee_gwei"].rolling(window=288, min_periods=12).std()
df_raw["fee_zscore"] = (df_raw["base_fee_gwei"] - rolling_mean_eval) / (rolling_std_eval + 1e-10)
df_raw["is_spike"] = df_raw["fee_zscore"] > SPIKE_Z_THRESHOLD

FEATURES = ["hour", "day_of_week", "fee_lag_1", "fee_rolling_mean", "fee_rolling_std", "fee_rolling_p75"]
if has_ratio:
    FEATURES.extend(["ratio_lag_1", "ratio_rolling_mean", "ratio_rolling_std"])

TARGET_FEE = "base_fee_gwei"
TARGET_RATIO = "gas_used_ratio"

cutoff = df_raw["timestamp"].max() - timedelta(hours=TEST_HOURS)
train_mask = df_raw["timestamp"] <= cutoff
test_mask  = df_raw["timestamp"] >  cutoff

X_train, y_train_fee = df_raw.loc[train_mask, FEATURES], df_raw.loc[train_mask, TARGET_FEE]
X_test, y_test_fee   = df_raw.loc[test_mask, FEATURES], df_raw.loc[test_mask, TARGET_FEE]

log(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

log("\nTraining LightGBM Fee Models (Quantile Regression)...")

def train_lgbm_quantile(X, y, alpha, is_fee_model=False, **kwargs):
    params = LGBM_PARAMS.copy()
    params.update({"objective": "quantile", "alpha": alpha})
    params.update(kwargs)
    model = lgb.LGBMRegressor(**params)
    
    if is_fee_model:
        weights = np.where(y > (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE), 50.0, 1.0)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
        
    return model

model_lower  = train_lgbm_quantile(X_train, y_train_fee, alpha=0.1, is_fee_model=True)
log("  [✓] Lower Bound Model (P10) Trained")

model_median = train_lgbm_quantile(X_train, y_train_fee, alpha=0.5, is_fee_model=True)
log("  [✓] Median Model (P50) Trained")

model_upper  = train_lgbm_quantile(X_train, y_train_fee, alpha=0.9, is_fee_model=True)
log("  [✓] Upper Bound Model (P90) Trained")

yhat_lower  = np.clip(model_lower.predict(X_test), BASE_L2_FLOOR_GWEI, None)
yhat_median = np.clip(model_median.predict(X_test), BASE_L2_FLOOR_GWEI, None)
yhat_upper  = np.clip(model_upper.predict(X_test), BASE_L2_FLOOR_GWEI, None)

is_spike_actual = df_raw.loc[test_mask, "is_spike"].values
m_all = calculate_metrics(y_test_fee.values, yhat_median, is_spike_actual)
ci_coverage = float(np.mean((y_test_fee.values >= yhat_lower) & (y_test_fee.values <= yhat_upper)) * 100)

log("\n" + "=" * 56)
log("  EVALUASI MODEL LIGHTGBM — Base L2 Optimized")
log("=" * 56)
log(f"  MAE           = {m_all['mae']:.8f} Gwei")
log(f"  RMSE          = {m_all['rmse']:.8f} Gwei")
log(f"  MAPE* = {m_all['mape']:.2f}%  (* spike periods only)")
log(f"  CI Coverage   = {ci_coverage:.1f}%")
log(f"  Precision     = {m_all.get('precision', 0):.2f}")
log(f"  Recall        = {m_all.get('recall', 0):.2f}")
log(f"  F1-Score      = {m_all.get('f1', 0):.2f}")
log("=" * 56)

if has_ratio:
    log("\nTraining LightGBM Ratio Model (Leading Indicator)...")
    y_train_ratio = df_raw.loc[train_mask, TARGET_RATIO]
    model_ratio = lgb.LGBMRegressor(**{**LGBM_PARAMS, "objective": "regression"})
    model_ratio.fit(X_train, y_train_ratio)
    log("  [✓] Ratio Model Trained")

log("\nRetraining pada FULL data untuk production...")
X_full = df_raw[FEATURES]
y_full_fee = df_raw[TARGET_FEE]

final_lower  = train_lgbm_quantile(X_full, y_full_fee, alpha=0.1, is_fee_model=True)
final_median = train_lgbm_quantile(X_full, y_full_fee, alpha=0.5, is_fee_model=True)
final_upper  = train_lgbm_quantile(X_full, y_full_fee, alpha=0.9, is_fee_model=True)

os.makedirs(MODEL_DIR, exist_ok=True)
final_lower.booster_.save_model(FEE_MODEL_LOWER)
final_median.booster_.save_model(FEE_MODEL_MEDIAN)
final_upper.booster_.save_model(FEE_MODEL_UPPER)

if has_ratio:
    y_full_ratio = df_raw[TARGET_RATIO]
    final_ratio = lgb.LGBMRegressor(**{**LGBM_PARAMS, "objective": "regression"})
    final_ratio.fit(X_full, y_full_ratio)
    final_ratio.booster_.save_model(RATIO_MODEL_PATH)

importance = pd.DataFrame({'feature': FEATURES, 'importance': final_median.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
log(f"\nFeature Importances:\n{importance.to_string(index=False)}")

report = {
    "trained_at": datetime.utcnow().isoformat(),
    "model_version": datetime.utcnow().strftime("v%Y%m%d_%H%M"),
    "data_path": DATA_PATH,
    "architecture": "LightGBM Quantile Regression",
    "features": FEATURES,
    "metrics_overall": m_all,
    "ci_coverage_pct": ci_coverage,
    "base_l2_floor_gwei": BASE_L2_FLOOR_GWEI,
    "floor_tolerance": FLOOR_TOLERANCE,
    "spike_z_threshold": SPIKE_Z_THRESHOLD,
    "rolling_window_hours": ROLLING_WINDOW_HOURS,
    "percentiles": {
        "mean": float(df_raw["base_fee_gwei"].mean()),
        "std": float(df_raw["base_fee_gwei"].std()),
        "p75": float(df_raw["base_fee_gwei"].quantile(0.75)),
        "p90": float(df_raw["base_fee_gwei"].quantile(0.90)),
        "p99": float(df_raw["base_fee_gwei"].quantile(0.99))
    }
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

log("\nTraining selesai. Pipeline siap untuk production.")