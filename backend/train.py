import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from prophet.diagnostics import cross_validation, performance_metrics
import logging
import warnings

logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

DATA_PATH         = "model/gas_fee_historical.csv"
MODEL_DIR         = "model"
FEE_MODEL_PATH    = f"{MODEL_DIR}/prophet_fee_model.json"
RATIO_MODEL_PATH  = f"{MODEL_DIR}/prophet_ratio_model.json"
REPORT_PATH       = f"{MODEL_DIR}/training_report.json"

TEST_HOURS        = 24
MIN_TRAIN_DAYS    = 3
MAPE_THRESHOLD    = 25.0

BASE_L2_FLOOR_GWEI = 0.005
FLOOR_TOLERANCE    = 0.0001
SPIKE_Z_THRESHOLD  = 2.0

ROLLING_WINDOW_HOURS = 6

FEE_PROPHET_CONFIG = {
    "changepoint_prior_scale":  0.15,
    "seasonality_mode":         "additive",
    "yearly_seasonality":       False,
    "weekly_seasonality":       True,
    "daily_seasonality":        True,
    "interval_width":           0.80,
    "uncertainty_samples":      500,
}

RATIO_PROPHET_CONFIG = {
    "changepoint_prior_scale":  0.05,
    "seasonality_mode":         "multiplicative",
    "yearly_seasonality":       False,
    "weekly_seasonality":       True,
    "daily_seasonality":        True,
    "interval_width":           0.80,
    "uncertainty_samples":      200,
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
            
        spike_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "spike_mae": spike_mae
        }
        
    return {"mae": mae, "rmse": rmse, "mape": mape, "within_10pct": within_10pct, **spike_metrics}

def calculate_baselines(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    actuals = df_test["base_fee_gwei"].values
    persistence_pred = np.full_like(actuals, df_train["base_fee_gwei"].iloc[-1])
    sma6h = df_train["base_fee_gwei"].tail(72).mean()
    sma_pred = np.full_like(actuals, sma6h)
    mean_pred = np.full_like(actuals, df_train["base_fee_gwei"].mean())
    
    return {
        "persistence_mae": float(mean_absolute_error(actuals, persistence_pred)),
        "sma6h_mae": float(mean_absolute_error(actuals, sma_pred)),
        "global_mean_mae": float(mean_absolute_error(actuals, mean_pred))
    }

def compute_z_scores(series: pd.Series, window: int = 288) -> pd.Series:

    rolling_mean = series.rolling(window=window, min_periods=12).mean()
    rolling_std  = series.rolling(window=window, min_periods=12).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)

def compute_savings_simulation(df_eval: pd.DataFrame, top_n_pct: float = 0.20) -> dict:

    window_size = 36
    actual_fees = df_eval["y"].values
    n = len(actual_fees)

    savings_list = []
    for i in range(n - window_size):
        current_fee  = actual_fees[i]
        future_min   = actual_fees[i:i + window_size].min()
        if current_fee > 0:
            saving_pct = (current_fee - future_min) / current_fee * 100
            savings_list.append(max(0, saving_pct))

    savings_arr = np.array(savings_list)

    spike_mask = actual_fees[:len(savings_arr)] > (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)

    return {
        "avg_savings_pct_all":   float(np.mean(savings_arr)),
        "avg_savings_pct_spike": float(np.mean(savings_arr[spike_mask[:len(savings_arr)]])) if spike_mask.sum() > 0 else 0.0,
        "max_savings_pct":       float(np.max(savings_arr)) if len(savings_arr) > 0 else 0.0,
        "spike_frequency_pct":   float(spike_mask.mean() * 100),
        "n_spike_periods":       int(spike_mask.sum()),
    }

def compute_ci_coverage(df_eval: pd.DataFrame) -> float:

    within = ((df_eval["y"] >= df_eval["yhat_lower"]) &
              (df_eval["y"] <= df_eval["yhat_upper"])).mean()
    return float(within * 100)

log(f"Loading data dari {DATA_PATH}...")
assert os.path.exists(DATA_PATH), f"File tidak ditemukan: {DATA_PATH}"

df_raw = pd.read_csv(DATA_PATH)
assert "timestamp"    in df_raw.columns, "Kolom 'timestamp' tidak ada"
assert "base_fee_gwei" in df_raw.columns, "Kolom 'base_fee_gwei' tidak ada"

df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True).dt.tz_localize(None)
df_raw = df_raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
df_raw = df_raw[(df_raw["base_fee_gwei"] > 0) & (df_raw["base_fee_gwei"] < 1000)]

has_ratio = "gas_used_ratio" in df_raw.columns

total_days = (df_raw["timestamp"].max() - df_raw["timestamp"].min()).days
log(f"Loaded: {len(df_raw):,} rows | {total_days} hari | {df_raw['timestamp'].min().date()} → {df_raw['timestamp'].max().date()}")
log(f"Fee stats: min={df_raw['base_fee_gwei'].min():.6f} mean={df_raw['base_fee_gwei'].mean():.6f} max={df_raw['base_fee_gwei'].max():.6f} Gwei")

assert total_days >= MIN_TRAIN_DAYS, f"Data terlalu sedikit: {total_days} hari (min {MIN_TRAIN_DAYS})"

log("Feature engineering...")

df_raw["fee_zscore"] = compute_z_scores(df_raw["base_fee_gwei"])
df_raw["is_spike"]   = df_raw["fee_zscore"] > SPIKE_Z_THRESHOLD
df_raw["is_floor"]   = df_raw["base_fee_gwei"] <= (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)

n_spikes = df_raw["is_spike"].sum()
n_floor  = df_raw["is_floor"].sum()
spike_pct = n_spikes / len(df_raw) * 100
floor_pct = n_floor  / len(df_raw) * 100

log(f"  Floor periods: {n_floor:,} ({floor_pct:.1f}%)")
log(f"  Spike periods: {n_spikes:,} ({spike_pct:.1f}%)")

ROLLING_W = int(ROLLING_WINDOW_HOURS * 12)
df_raw["fee_rolling_mean"] = df_raw["base_fee_gwei"].rolling(ROLLING_W, min_periods=6).mean()
df_raw["fee_rolling_std"]  = df_raw["base_fee_gwei"].rolling(ROLLING_W, min_periods=6).std()
df_raw["fee_rolling_p25"]  = df_raw["base_fee_gwei"].rolling(ROLLING_W, min_periods=6).quantile(0.25)
df_raw["fee_rolling_p75"]  = df_raw["base_fee_gwei"].rolling(ROLLING_W, min_periods=6).quantile(0.75)

if has_ratio:
    df_raw["ratio_rolling_mean"] = df_raw["gas_used_ratio"].rolling(ROLLING_W, min_periods=6).mean()
    df_raw["ratio_rolling_std"]  = df_raw["gas_used_ratio"].rolling(ROLLING_W, min_periods=6).std()

df_raw = df_raw.dropna(subset=["fee_rolling_mean"]).reset_index(drop=True)
log(f"  After rolling feature drop: {len(df_raw):,} rows")

cutoff = df_raw["timestamp"].max() - timedelta(hours=TEST_HOURS)
df_train = df_raw[df_raw["timestamp"] <= cutoff].copy()
df_test  = df_raw[df_raw["timestamp"] >  cutoff].copy()

log(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

global_p25   = float(df_train["base_fee_gwei"].quantile(0.25))
global_p75   = float(df_train["base_fee_gwei"].quantile(0.75))
global_p90   = float(df_train["base_fee_gwei"].quantile(0.90))
global_p95   = float(df_train["base_fee_gwei"].quantile(0.95))
global_p99   = float(df_train["base_fee_gwei"].quantile(0.99))
global_mean  = float(df_train["base_fee_gwei"].mean())
global_std   = float(df_train["base_fee_gwei"].std())

log(f"Global: P25={global_p25:.6f} P75={global_p75:.6f} P90={global_p90:.6f} P99={global_p99:.6f} Gwei")

if has_ratio:
    ratio_p25  = float(df_train["gas_used_ratio"].quantile(0.25))
    ratio_p50  = float(df_train["gas_used_ratio"].quantile(0.50))
    ratio_p75  = float(df_train["gas_used_ratio"].quantile(0.75))
    ratio_p90  = float(df_train["gas_used_ratio"].quantile(0.90))
    log(f"Ratio:  P25={ratio_p25:.4f} P50={ratio_p50:.4f} P75={ratio_p75:.4f} P90={ratio_p90:.4f}")

log("\nTraining fee Prophet model...")

df_fee_train = pd.DataFrame({
    "ds": df_train["timestamp"],
    "y":  df_train["base_fee_gwei"],
})
if has_ratio:
    df_fee_train["gas_used_ratio"] = df_train["gas_used_ratio"].fillna(df_train["gas_used_ratio"].median())

fee_eval_model = Prophet(**FEE_PROPHET_CONFIG)
if has_ratio:
    fee_eval_model.add_regressor("gas_used_ratio", standardize=True)
fee_eval_model.fit(df_fee_train)

df_fee_test = pd.DataFrame({"ds": df_test["timestamp"]})
if has_ratio:
    df_fee_test["gas_used_ratio"] = df_test["gas_used_ratio"].fillna(df_test["gas_used_ratio"].median()).values

fc_fee = fee_eval_model.predict(df_fee_test)
fc_fee["yhat"]       = fc_fee["yhat"].clip(lower=0)
fc_fee["yhat_lower"] = fc_fee["yhat_lower"].clip(lower=0)
fc_fee["yhat_upper"] = fc_fee["yhat_upper"].clip(lower=0)

df_eval_fee = df_test.copy().reset_index(drop=True)
df_eval_fee["yhat"]       = fc_fee["yhat"].values
df_eval_fee["yhat_lower"] = fc_fee["yhat_lower"].values
df_eval_fee["yhat_upper"] = fc_fee["yhat_upper"].values
df_eval_fee.rename(columns={"base_fee_gwei": "y"}, inplace=True)

ratio_metrics = None
if has_ratio:
    log("Training gas_used_ratio Prophet model (leading indicator)...")

    df_ratio_train = pd.DataFrame({
        "ds": df_train["timestamp"],
        "y":  df_train["gas_used_ratio"].fillna(df_train["gas_used_ratio"].median()),
    })

    ratio_eval_model = Prophet(**RATIO_PROPHET_CONFIG)
    ratio_eval_model.fit(df_ratio_train)

    df_ratio_test = pd.DataFrame({"ds": df_test["timestamp"]})
    fc_ratio = ratio_eval_model.predict(df_ratio_test)
    fc_ratio["yhat"] = fc_ratio["yhat"].clip(lower=0, upper=1)

    actual_ratio  = df_test["gas_used_ratio"].fillna(df_test["gas_used_ratio"].median()).values
    pred_ratio    = fc_ratio["yhat"].values
    ratio_metrics = calculate_metrics(actual_ratio, pred_ratio)

    log(f"  Ratio MAE={ratio_metrics['mae']:.4f} RMSE={ratio_metrics['rmse']:.4f} MAPE={ratio_metrics['mape']:.2f}%")
else:
    log("  gas_used_ratio tidak tersedia → skip ratio model")

log("\n" + "=" * 56)
log("  EVALUASI MODEL — Base L2 Optimized")
log("=" * 56)

is_spike_actual = df_eval_fee["y"].values > (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)
m_all = calculate_metrics(df_eval_fee["y"].values, df_eval_fee["yhat"].values, is_spike_actual)
ci_coverage = compute_ci_coverage(df_eval_fee)
savings_sim = compute_savings_simulation(df_eval_fee)
baselines = calculate_baselines(df_train, df_test)

log(f"\n  Baseline Comparisons (MAE):")
log(f"  Persistence MAE = {baselines['persistence_mae']:.8f}")
log(f"  SMA 6h MAE      = {baselines['sma6h_mae']:.8f}")
log(f"  Global Mean MAE = {baselines['global_mean_mae']:.8f}")

log(f"\n  Fee Model (24h test):")
log(f"  MAE           = {m_all['mae']:.8f} Gwei")
log(f"  RMSE          = {m_all['rmse']:.8f} Gwei")
log(f"  MAPE*         = {m_all['mape']:.2f}%  (* spike periods only)")
log(f"  Within 10%    = {m_all['within_10pct']:.1f}% of predictions")
log(f"  CI Coverage   = {ci_coverage:.1f}%  (target: ~80%)")

log(f"\n  Spike Detection Metrics:")
log(f"  Precision     = {m_all.get('precision', 0):.2f}")
log(f"  Recall        = {m_all.get('recall', 0):.2f}")
log(f"  F1-Score      = {m_all.get('f1', 0):.2f}")
log(f"  Spike MAE     = {m_all.get('spike_mae', 0):.8f}")

log(f"\n  Savings Simulation:")
log(f"  Avg saving (all)    = {savings_sim['avg_savings_pct_all']:.2f}%")
log(f"  Avg saving (spike)  = {savings_sim['avg_savings_pct_spike']:.2f}%")
log(f"  Max saving          = {savings_sim['max_savings_pct']:.2f}%")
log(f"  Spike frequency     = {savings_sim['spike_frequency_pct']:.2f}% of time")

horizon_metrics = {}
for hours in [1, 2, 3]:
    n = hours * 12
    sub = df_eval_fee.head(n)
    sub_spike = is_spike_actual[:n]
    m   = calculate_metrics(sub["y"].values, sub["yhat"].values, sub_spike)
    horizon_metrics[f"{hours}h"] = m
    log(f"\n  {hours}h horizon: MAE={m['mae']:.8f} MAPE={m['mape']:.2f}% F1={m.get('f1',0):.2f}")

log("=" * 56 + "\n")

log("Menjalankan Ablation Study (Tanpa Gas Ratio Regressor)...")
ablation_model = Prophet(**FEE_PROPHET_CONFIG)
ablation_model.fit(pd.DataFrame({"ds": df_train["timestamp"], "y": df_train["base_fee_gwei"]}))
fc_ablation = ablation_model.predict(pd.DataFrame({"ds": df_test["timestamp"]}))
fc_ablation["yhat"] = fc_ablation["yhat"].clip(lower=0)
m_ablation = calculate_metrics(df_test["base_fee_gwei"].values, fc_ablation["yhat"].values, is_spike_actual)
log(f"  Tanpa Regressor -> MAE={m_ablation['mae']:.8f}, F1={m_ablation.get('f1',0):.2f}")

log("\nMenjalankan Time-Series Cross Validation (Expanding Window)...")
cv_results = cross_validation(ablation_model, initial=f'{MIN_TRAIN_DAYS} days', period='1 days', horizon='1 days')
cv_metrics = performance_metrics(cv_results)
cv_mae_mean = cv_metrics['mae'].mean()
log(f"  CV MAE Mean (1-day horizon) = {cv_mae_mean:.8f}")

log("Retraining final models pada FULL data...")

df_fee_full = pd.DataFrame({
    "ds": df_raw["timestamp"],
    "y":  df_raw["base_fee_gwei"],
})
if has_ratio:
    df_fee_full["gas_used_ratio"] = df_raw["gas_used_ratio"].fillna(df_raw["gas_used_ratio"].median())

final_fee_model = Prophet(**FEE_PROPHET_CONFIG)
if has_ratio:
    final_fee_model.add_regressor("gas_used_ratio", standardize=True)
final_fee_model.fit(df_fee_full)
log("  Fee model: trained")

if has_ratio:
    df_ratio_full = pd.DataFrame({
        "ds": df_raw["timestamp"],
        "y":  df_raw["gas_used_ratio"].fillna(df_raw["gas_used_ratio"].median()),
    })
    final_ratio_model = Prophet(**RATIO_PROPHET_CONFIG)
    final_ratio_model.fit(df_ratio_full)
    log("  Ratio model: trained")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(FEE_MODEL_PATH, "w") as f:
    f.write(model_to_json(final_fee_model))
log(f"  Fee model   → {FEE_MODEL_PATH}")

if has_ratio:
    with open(RATIO_MODEL_PATH, "w") as f:
        f.write(model_to_json(final_ratio_model))
    log(f"  Ratio model → {RATIO_MODEL_PATH}")

report = {
    "trained_at":               datetime.utcnow().isoformat(),
    "model_version":            datetime.utcnow().strftime("v%Y%m%d_%H%M"),
    "data_path":                DATA_PATH,
    "chain":                    "Base L2",
    "train_rows":               len(df_fee_train),
    "test_rows":                len(df_test),
    "train_period_start":       str(df_train["timestamp"].min()),
    "train_period_end":         str(df_train["timestamp"].max()),
    "has_gas_ratio_regressor":  has_ratio,
    "has_ratio_model":          has_ratio,
    "fee_prophet_config":       FEE_PROPHET_CONFIG,
    "ratio_prophet_config":     RATIO_PROPHET_CONFIG if has_ratio else None,

    "base_l2_floor_gwei":       BASE_L2_FLOOR_GWEI,
    "floor_tolerance":          FLOOR_TOLERANCE,
    "spike_z_threshold":        SPIKE_Z_THRESHOLD,
    "rolling_window_hours":     ROLLING_WINDOW_HOURS,

    "percentiles": {
        "p25": global_p25, "p50": float(df_train["base_fee_gwei"].median()),
        "p75": global_p75, "p90": global_p90, "p95": global_p95, "p99": global_p99,
        "mean": global_mean, "std": global_std,
    },

    "ratio_percentiles": {
        "p25": ratio_p25, "p50": ratio_p50, "p75": ratio_p75, "p90": ratio_p90,
    } if has_ratio else None,

    "metrics_overall":      m_all,
    "metrics_per_horizon":  horizon_metrics,
    "ci_coverage_pct":      ci_coverage,
    "mape_ok":              m_all["mape"] < MAPE_THRESHOLD,
    "ratio_metrics":        ratio_metrics,
    "baselines":            baselines,
    "ablation_metrics":     m_ablation,
    "cv_mae_mean":          cv_mae_mean,

    "savings_simulation":   savings_sim,

    "data_characteristics": {
        "floor_pct":       floor_pct,
        "spike_pct":       spike_pct,
        "n_spikes":        int(n_spikes),
        "fee_min":         float(df_raw["base_fee_gwei"].min()),
        "fee_max":         float(df_raw["base_fee_gwei"].max()),
        "fee_mean":        float(df_raw["base_fee_gwei"].mean()),
        "fee_std":         float(df_raw["base_fee_gwei"].std()),
    },
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
log(f"  Report → {REPORT_PATH}")

log("\nTraining selesai.")
log(f"   Fee model:   {FEE_MODEL_PATH}")
log(f"   Ratio model: {RATIO_MODEL_PATH}")
log(f"   Report:      {REPORT_PATH}")
log(f"   Jalankan:    uvicorn main:app --reload")
