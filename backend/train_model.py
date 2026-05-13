

import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DB_PATH = "model/gas_fee.db"
MODEL_DIR = "model"
REPORT_PATH = f"{MODEL_DIR}/training_report.json"

TEST_HOURS = 24
VAL_HOURS = 24
ROLLING_WINDOW = 72
SPIKE_Z_THRESH = 2.0
MAPE_THRESHOLD = 25.0
WALK_FORWARD_WINDOWS = 3
MIN_TRAIN_ROWS = 1500

LGBM_BASE = dict(
    boosting_type="gbdt",
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    verbose=-1,
    n_estimators=500,
    early_stopping_rounds=30,
)

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def build_features(df_sorted: pd.DataFrame) -> pd.DataFrame:
    
    df = df_sorted.copy()
    df["interval_ts"] = pd.to_datetime(df["interval_ts"])

    df["hour"] = df["interval_ts"].dt.hour
    df["day_of_week"] = df["interval_ts"].dt.dayofweek
    df["minute"] = df["interval_ts"].dt.minute

    lag_fee = df["base_fee_gwei"].shift(1)
    df["fee_lag_1"] = lag_fee
    df["fee_lag_2"] = df["base_fee_gwei"].shift(2)
    df["fee_lag_3"] = df["base_fee_gwei"].shift(3)
    df["fee_rolling_mean"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).mean()
    df["fee_rolling_std"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).std().fillna(0)
    df["fee_rolling_p25"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.25)
    df["fee_rolling_p75"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.75)
    df["fee_rolling_p90"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.90)

    df["fee_diff_1"] = df["fee_lag_1"] - df["fee_lag_2"]
    df["fee_diff_2"] = df["fee_lag_2"] - df["fee_lag_3"]

    df["fee_zscore"] = (df["fee_lag_1"] - df["fee_rolling_mean"]) / (df["fee_rolling_std"] + 1e-10)

    lag_ratio = df["gas_used_ratio"].shift(1)
    df["ratio_lag_1"] = lag_ratio
    df["ratio_rolling_mean"] = lag_ratio.rolling(ROLLING_WINDOW, min_periods=6).mean()
    df["ratio_rolling_std"] = lag_ratio.rolling(ROLLING_WINDOW, min_periods=6).std().fillna(0)
    df["ratio_diff_1"] = lag_ratio - df["gas_used_ratio"].shift(2)

    return df.dropna().reset_index(drop=True)

FEATURES = [
    "hour", "day_of_week", "minute",
    "fee_lag_1", "fee_lag_2", "fee_lag_3",
    "fee_rolling_mean", "fee_rolling_std",
    "fee_rolling_p25", "fee_rolling_p75", "fee_rolling_p90",
    "fee_diff_1", "fee_diff_2", "fee_zscore",
    "ratio_lag_1", "ratio_rolling_mean", "ratio_rolling_std", "ratio_diff_1",
]

def temporal_split(df: pd.DataFrame, end_ts: pd.Timestamp, test_h: int, val_h: int):
    
    ts = pd.to_datetime(df["interval_ts"])
    test_start = end_ts - timedelta(hours=test_h)
    val_start = end_ts - timedelta(hours=test_h + val_h)

    train_df = df[ts <= val_start].copy()
    val_df = df[(ts > val_start) & (ts <= test_start)].copy()
    test_df = df[ts > test_start].copy()
    return train_df, val_df, test_df

def effective_n_estimators(model: lgb.LGBMRegressor) -> int:
    
    bi = model.best_iteration_
    cap = int(LGBM_BASE["n_estimators"])
    if bi is None:
        return cap
    return int(min(cap, max(1, bi + 1)))

def train_quantile(
    X_tr, y_tr, X_val, y_val, alpha: float, label: str,
    floor: float, tol: float, quiet: bool = False,
) -> lgb.LGBMRegressor:
    params = {**LGBM_BASE, "objective": "quantile", "alpha": alpha}
    spike_mask = y_tr > (floor + tol)
    weights = np.where(spike_mask, 10.0, 1.0)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    if not quiet:
        log(f"  [] {label}  best iter: {model.best_iteration_}")
    return model

def train_ratio_model(X_tr, y_tr, X_val, y_val, label: str, quiet: bool = False) -> lgb.LGBMRegressor:
    params = {**LGBM_BASE, "objective": "regression"}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    if not quiet:
        log(f"  [] {label}  best iter: {model.best_iteration_}")
    return model

def fee_metrics_block(
    y_true: np.ndarray,
    y_pred_med: np.ndarray,
    y_pred_lo: np.ndarray,
    y_pred_hi: np.ndarray,
    floor: float,
    tol: float,
    spike_mask: np.ndarray,
) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred_med))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_med)))
    mask_nf = y_true > (floor + tol)
    mape = (
        float(np.mean(np.abs((y_true[mask_nf] - y_pred_med[mask_nf]) / (y_true[mask_nf] + 1e-10))) * 100)
        if mask_nf.sum() > 0 else 0.0
    )
    ci_cov = float(np.mean((y_true >= y_pred_lo) & (y_true <= y_pred_hi)) * 100)
    within_10 = float(np.mean(np.abs(y_true - y_pred_med) / (y_true + 1e-10) < 0.10) * 100)

    out = {
        "mae": mae,
        "rmse": rmse,
        "mape_non_floor_pct": mape,
        "ci_coverage_p10_p90_pct": ci_cov,
        "within_10pct_pct": within_10,
        "n": int(len(y_true)),
    }

    sm = np.asarray(spike_mask, dtype=bool)
    if sm.any():
        ys, ps = y_true[sm], y_pred_med[sm]
        out["spike_only"] = {
            "n": int(sm.sum()),
            "mae": float(mean_absolute_error(ys, ps)),
            "mape_non_floor_pct": float(
                np.mean(np.abs((ys - ps) / (ys + 1e-10))) * 100
            ),
        }
    else:
        out["spike_only"] = {"n": 0, "mae": None, "mape_non_floor_pct": None}

    nsm = ~sm
    if nsm.any():
        ys, ps = y_true[nsm], y_pred_med[nsm]
        out["non_spike"] = {
            "n": int(nsm.sum()),
            "mae": float(mean_absolute_error(ys, ps)),
        }
    else:
        out["non_spike"] = {"n": 0, "mae": None}

    return out

def predict_fee_bundle(models, X, floor: float):
    lo, med, up = models
    return (
        np.clip(lo.predict(X), floor, None),
        np.clip(med.predict(X), floor, None),
        np.clip(up.predict(X), floor, None),
    )

def run_training_pipeline(df: pd.DataFrame, floor: float, tol: float) -> dict:
    
    end_ts = pd.to_datetime(df["interval_ts"]).max()
    train_df, val_df, test_df = temporal_split(df, end_ts, TEST_HOURS, VAL_HOURS)

    if len(train_df) < MIN_TRAIN_ROWS or len(val_df) < 12 or len(test_df) < 12:
        raise RuntimeError(
            f"Data kurang untuk split {TEST_HOURS}h/{VAL_HOURS}h: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    X_tr, y_tr = train_df[FEATURES], train_df["base_fee_gwei"]
    X_va, y_va = val_df[FEATURES], val_df["base_fee_gwei"]
    X_te, y_te = test_df[FEATURES], test_df["base_fee_gwei"].values

    log(f"Split (ujung={end_ts}): train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,}")

    log("\nTraining quantile models (ES on val, test untouched)...")
    m_lo = train_quantile(X_tr, y_tr, X_va, y_va, 0.10, "Lower  (P10)", floor, tol)
    m_md = train_quantile(X_tr, y_tr, X_va, y_va, 0.50, "Median (P50)", floor, tol)
    m_hi = train_quantile(X_tr, y_tr, X_va, y_va, 0.90, "Upper  (P90)", floor, tol)

    log("Training ratio model...")
    m_ratio = train_ratio_model(
        X_tr, train_df["gas_used_ratio"],
        X_va, val_df["gas_used_ratio"],
        "Ratio model",
    )

    y_plo, y_pmd, y_pup = predict_fee_bundle((m_lo, m_md, m_hi), X_te, floor)
    spike_te = test_df["is_spike"].values

    metrics = fee_metrics_block(y_te, y_pmd, y_plo, y_pup, floor, tol, spike_te)
    metrics["r2_median_oos"] = float(r2_score(y_te, y_pmd))
    y_ratio_true = test_df["gas_used_ratio"].values
    y_ratio_pred = m_ratio.predict(X_te)
    metrics["ratio_oos"] = {
        "mae": float(mean_absolute_error(y_ratio_true, y_ratio_pred)),
        "n": int(len(y_ratio_true)),
    }

    y_base = test_df["fee_lag_1"].values
    mae_base = float(mean_absolute_error(y_te, y_base))
    mask_nf = y_te > (floor + tol)
    mape_base = (
        float(np.mean(np.abs((y_te[mask_nf] - y_base[mask_nf]) / (y_te[mask_nf] + 1e-10))) * 100)
        if mask_nf.sum() > 0 else 0.0
    )
    metrics["baseline_lag1"] = {
        "mae": mae_base,
        "mape_non_floor_pct": mape_base,
        "mae_improvement_vs_model": float(mae_base - metrics["mae"]),
    }

    return {
        "end_ts": end_ts,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "models": (m_lo, m_md, m_hi, m_ratio),
        "metrics_primary_oos": metrics,
        "y_test": y_te,
        "y_pred_median": y_pmd,
    }

def walk_forward_eval(df: pd.DataFrame, floor: float, tol: float) -> list:
    
    ts_max = pd.to_datetime(df["interval_ts"]).max()
    rows = []
    for w in range(WALK_FORWARD_WINDOWS):
        end_ts = ts_max - timedelta(hours=w * TEST_HOURS)
        train_df, val_df, test_df = temporal_split(df, end_ts, TEST_HOURS, VAL_HOURS)
        if len(train_df) < MIN_TRAIN_ROWS or len(val_df) < 12 or len(test_df) < 12:
            continue
        X_tr, y_tr = train_df[FEATURES], train_df["base_fee_gwei"]
        X_va, y_va = val_df[FEATURES], val_df["base_fee_gwei"]
        X_te, y_te = test_df[FEATURES], test_df["base_fee_gwei"].values

        m_md = train_quantile(X_tr, y_tr, X_va, y_va, 0.50, f"WF P50 w{w}", floor, tol, quiet=True)
        y_pmd = np.clip(m_md.predict(X_te), floor, None)
        mae = float(mean_absolute_error(y_te, y_pmd))
        mae_lag = float(mean_absolute_error(y_te, test_df["fee_lag_1"].values))
        rows.append({
            "window": w,
            "test_end_ts": str(end_ts),
            "mae_median_model": mae,
            "mae_baseline_lag1": mae_lag,
            "n_test": len(y_te),
        })
    return rows

def refit_production(
    df: pd.DataFrame,
    floor: float,
    tol: float,
    m_lo, m_md, m_hi, m_ratio,
):
    
    X_full = df[FEATURES]
    y_fee = df["base_fee_gwei"]
    y_ratio = df["gas_used_ratio"]
    w_fee = np.where(y_fee > (floor + tol), 10.0, 1.0)

    def one_q(m, alpha):
        p = {**LGBM_BASE, "objective": "quantile", "alpha": alpha}
        p.pop("early_stopping_rounds", None)
        p["n_estimators"] = effective_n_estimators(m)
        out = lgb.LGBMRegressor(**p)
        out.fit(X_full, y_fee, sample_weight=w_fee)
        return out

    f_lo = one_q(m_lo, 0.10)
    f_md = one_q(m_md, 0.50)
    f_hi = one_q(m_hi, 0.90)

    pr = {**LGBM_BASE, "objective": "regression"}
    pr.pop("early_stopping_rounds", None)
    pr["n_estimators"] = effective_n_estimators(m_ratio)
    f_ratio = lgb.LGBMRegressor(**pr)
    f_ratio.fit(X_full, y_ratio)

    return f_lo, f_md, f_hi, f_ratio

def main():
    log(f"Loading data dari {DB_PATH}...")
    assert os.path.exists(DB_PATH), f"DB tidak ditemukan: {DB_PATH}"

    con = sqlite3.connect(DB_PATH)
    df_raw = pd.read_sql_query(
        "SELECT interval_ts, base_fee_gwei, gas_used_ratio FROM gas_fee ORDER BY interval_ts",
        con,
    )
    con.close()

    log(f"Raw rows: {len(df_raw):,}")
    df_raw = df_raw[(df_raw["base_fee_gwei"] > 0) & (df_raw["base_fee_gwei"] < 1000)]

    floor = float(np.percentile(df_raw["base_fee_gwei"], 1))
    tol = float(df_raw["base_fee_gwei"].std() * 0.1)
    log(f"Floor terdeteksi dari data: {floor:.6f} Gwei (tolerance {tol:.6f})")

    log("Feature engineering...")
    df = build_features(df_raw)
    log(f"Setelah feature engineering: {len(df):,} rows")

    rolling_mean = df["fee_lag_1"].rolling(288, min_periods=12).mean()
    rolling_std = df["fee_lag_1"].rolling(288, min_periods=12).std()
    df["is_spike"] = ((df["base_fee_gwei"] - rolling_mean) / (rolling_std + 1e-10)) > SPIKE_Z_THRESH

    result = run_training_pipeline(df, floor, tol)
    m_lo, m_md, m_hi, m_ratio = result["models"]
    metrics = result["metrics_primary_oos"]

    log(f"\n{'='*52}")
    log(f"  OOS test (24h terakhir, tidak dipakai ES / train)")
    log(f"  MAE (model)   = {metrics['mae']:.8f} Gwei")
    log(f"  MAE (lag-1)   = {metrics['baseline_lag1']['mae']:.8f} Gwei")
    log(f"   MAE (lagmodel) = {metrics['baseline_lag1']['mae_improvement_vs_model']:.8f}  (>0 = model lebih baik)")
    log(f"  RMSE          = {metrics['rmse']:.8f} Gwei")
    log(f"  MAPE*         = {metrics['mape_non_floor_pct']:.2f}%  (* non-floor)")
    log(f"  CI P10P90    = {metrics['ci_coverage_p10_p90_pct']:.1f}%  (nominal ~80%)")
    log(f"  Within 10%    = {metrics['within_10pct_pct']:.1f}%")
    log(f"  Ratio MAE OOS = {metrics['ratio_oos']['mae']:.6f}")
    sp = metrics["spike_only"]
    if sp["n"] > 0:
        log(f"  Spike-only MAE = {sp['mae']:.8f}  (n={sp['n']})")
    else:
        log("  Spike-only MAE = n/a (tidak ada baris spike di test OOS)")
    log(f"  R (median OOS) = {metrics['r2_median_oos']:.6f}")
    log(f"{'='*52}")

    log("\nWalk-forward (median fee saja, beberapa window)...")
    wf = walk_forward_eval(df, floor, tol)
    for r in wf:
        log(
            f"  wf{r['window']}: MAE_model={r['mae_median_model']:.8f} "
            f"MAE_lag1={r['mae_baseline_lag1']:.8f} n={r['n_test']}"
        )

    log("\nRetrain production (full data, n_trees dari ES)...")
    f_lo, f_md, f_hi, f_ratio = refit_production(df, floor, tol, m_lo, m_md, m_hi, m_ratio)

    os.makedirs(MODEL_DIR, exist_ok=True)
    f_lo.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_lower.txt")
    f_md.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_median.txt")
    f_hi.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_upper.txt")
    f_ratio.booster_.save_model(f"{MODEL_DIR}/lgbm_ratio_median.txt")
    log("Model tersimpan.")

    imp = pd.DataFrame({"feature": FEATURES, "importance": f_md.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    log(f"\nFeature Importances (top 10):\n{imp.head(10).to_string(index=False)}")

    feat_imp_top = [
        {"feature": str(r["feature"]), "importance": int(r["importance"])}
        for _, r in imp.head(14).iterrows()
    ]

    ci = metrics["ci_coverage_p10_p90_pct"]
    nominal = 80.0
    report = {
        "trained_at": datetime.utcnow().isoformat(),
        "model_version": datetime.utcnow().strftime("v%Y%m%d_%H%M"),
        "architecture": "LightGBM Quantile Regression",
        "evaluation_protocol": {
            "split": f"train | val={VAL_HOURS}h (early stopping only) | test={TEST_HOURS}h (OOS metrics)",
            "production_refit": "all rows; n_estimators per model = best_iteration+1 from train+val ES",
            "ci_nominal_central_pct": nominal,
            "ci_calibration_note": "Ideal P10P90 coverage ~80%. Jauh di atas = interval terlalu lebar (konservatif).",
        },
        "features": FEATURES,
        "rolling_window": ROLLING_WINDOW,
        "base_l2_floor_gwei": floor,
        "floor_tolerance": tol,
        "spike_z_threshold": SPIKE_Z_THRESH,
        "mape_ok": metrics["mape_non_floor_pct"] < MAPE_THRESHOLD,
        "metrics_oos_test": metrics,
        "metrics_overall": {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape_non_floor_pct"],
            "within_10pct": metrics["within_10pct_pct"],
            "r2": metrics["r2_median_oos"],
        },
        "feature_importance_top": feat_imp_top,
        "ci_coverage_pct": metrics["ci_coverage_p10_p90_pct"],
        "walk_forward_median_fee": wf,
        "percentiles": {
            "p1": float(df["base_fee_gwei"].quantile(0.01)),
            "p25": float(df["base_fee_gwei"].quantile(0.25)),
            "mean": float(df["base_fee_gwei"].mean()),
            "std": float(df["base_fee_gwei"].std()),
            "p75": float(df["base_fee_gwei"].quantile(0.75)),
            "p90": float(df["base_fee_gwei"].quantile(0.90)),
            "p99": float(df["base_fee_gwei"].quantile(0.99)),
        },
        "production_trees": {
            "fee_lower": effective_n_estimators(m_lo),
            "fee_median": effective_n_estimators(m_md),
            "fee_upper": effective_n_estimators(m_hi),
            "ratio": effective_n_estimators(m_ratio),
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nReport disimpan ke {REPORT_PATH}")
    log("Training pipeline selesai.")

if __name__ == "__main__":
    main()
