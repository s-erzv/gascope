
import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

DB_PATH      = "model/gas_fee.db"
MODEL_DIR    = "model"
REPORT_PATH  = f"{MODEL_DIR}/training_report.json"
PROFILES_PATH = f"{MODEL_DIR}/hourly_profiles.json"

TEST_HOURS          = 24
VAL_HOURS           = 24
ROLLING_WINDOW      = 72
SPIKE_Z_THRESH      = 2.0
MAPE_THRESHOLD      = 25.0
WALK_FORWARD_WINDOWS = 3
MIN_TRAIN_ROWS      = 1500
SPIKE_HORIZON       = 12   # intervals = 1 hour

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


# ──────────────────────────────────────────────
# Hourly profiles (computed before feature eng.)
# ──────────────────────────────────────────────

def build_hourly_profiles(df_raw: pd.DataFrame, floor: float, tol: float) -> dict:
    df = df_raw.copy()
    df["_hour"] = pd.to_datetime(df["interval_ts"]).dt.hour

    rolling_mean = df["base_fee_gwei"].rolling(288, min_periods=12).mean()
    rolling_std  = df["base_fee_gwei"].rolling(288, min_periods=12).std().fillna(0)
    df["_is_spike"] = ((df["base_fee_gwei"] - rolling_mean) / (rolling_std + 1e-10)) > SPIKE_Z_THRESH

    profiles: dict = {}
    for h in range(24):
        mask = df["_hour"] == h
        sub  = df[mask]
        if len(sub) == 0:
            profiles[h] = {"mean_fee": float(floor), "spike_rate": 0.0, "mean_ratio": 0.5}
        else:
            profiles[h] = {
                "mean_fee":   float(sub["base_fee_gwei"].mean()),
                "spike_rate": float(sub["_is_spike"].mean()),
                "mean_ratio": float(sub["gas_used_ratio"].mean()),
            }
    return profiles


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

FEATURES = [
    "hour", "day_of_week", "minute",
    # Short lags (5–15 min)
    "fee_lag_1", "fee_lag_2", "fee_lag_3",
    # Medium lags (30 min, 1h, 2h)
    "fee_lag_6", "fee_lag_12", "fee_lag_24",
    # Rolling statistics
    "fee_rolling_mean", "fee_rolling_std",
    "fee_rolling_p25", "fee_rolling_p75", "fee_rolling_p90",
    # Momentum
    "fee_diff_1", "fee_diff_2",
    # Floor-relative features
    "fee_above_floor", "is_at_floor",
    # Normalised deviation
    "fee_zscore",
    # Gas utilisation
    "ratio_lag_1", "ratio_rolling_mean", "ratio_rolling_std", "ratio_diff_1",
    # Seasonality
    "hour_spike_rate",
]


def build_features(
    df_sorted: pd.DataFrame,
    floor: float,
    tol: float,
    hour_spike_rates: Optional[dict] = None,
) -> pd.DataFrame:
    df = df_sorted.copy()
    df["interval_ts"] = pd.to_datetime(df["interval_ts"])

    df["hour"]        = df["interval_ts"].dt.hour
    df["day_of_week"] = df["interval_ts"].dt.dayofweek
    df["minute"]      = df["interval_ts"].dt.minute

    lag_fee = df["base_fee_gwei"].shift(1)
    df["fee_lag_1"]  = lag_fee
    df["fee_lag_2"]  = df["base_fee_gwei"].shift(2)
    df["fee_lag_3"]  = df["base_fee_gwei"].shift(3)
    df["fee_lag_6"]  = df["base_fee_gwei"].shift(6)
    df["fee_lag_12"] = df["base_fee_gwei"].shift(12)
    df["fee_lag_24"] = df["base_fee_gwei"].shift(24)

    df["fee_rolling_mean"] = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).mean()
    df["fee_rolling_std"]  = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).std().fillna(0)
    df["fee_rolling_p25"]  = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.25)
    df["fee_rolling_p75"]  = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.75)
    df["fee_rolling_p90"]  = lag_fee.rolling(ROLLING_WINDOW, min_periods=6).quantile(0.90)

    df["fee_diff_1"] = df["fee_lag_1"] - df["fee_lag_2"]
    df["fee_diff_2"] = df["fee_lag_2"] - df["fee_lag_3"]
    df["fee_zscore"] = (df["fee_lag_1"] - df["fee_rolling_mean"]) / (df["fee_rolling_std"] + 1e-10)

    df["fee_above_floor"] = (df["fee_lag_1"] - floor).clip(lower=0.0)
    df["is_at_floor"]     = (df["fee_lag_1"] <= floor + tol).astype(float)

    if hour_spike_rates is not None:
        df["hour_spike_rate"] = df["hour"].map(hour_spike_rates).fillna(0.0)
    else:
        df["hour_spike_rate"] = 0.0

    lag_ratio = df["gas_used_ratio"].shift(1)
    df["ratio_lag_1"]        = lag_ratio
    df["ratio_rolling_mean"] = lag_ratio.rolling(ROLLING_WINDOW, min_periods=6).mean()
    df["ratio_rolling_std"]  = lag_ratio.rolling(ROLLING_WINDOW, min_periods=6).std().fillna(0)
    df["ratio_diff_1"]       = lag_ratio - df["gas_used_ratio"].shift(2)

    return df.dropna().reset_index(drop=True)


# ──────────────────────────────────────────────
# Temporal split
# ──────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, end_ts: pd.Timestamp, test_h: int, val_h: int):
    ts         = pd.to_datetime(df["interval_ts"])
    test_start = end_ts - timedelta(hours=test_h)
    val_start  = end_ts - timedelta(hours=test_h + val_h)

    train_df = df[ts <= val_start].copy()
    val_df   = df[(ts > val_start) & (ts <= test_start)].copy()
    test_df  = df[ts > test_start].copy()
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# Model training helpers
# ──────────────────────────────────────────────

def effective_n_estimators(model) -> int:
    bi  = getattr(model, "best_iteration_", None)
    cap = int(LGBM_BASE["n_estimators"])
    if bi is None:
        return cap
    return int(min(cap, max(1, bi + 1)))


def train_quantile(
    X_tr, y_tr, X_val, y_val,
    alpha: float, label: str,
    floor: float, tol: float,
    quiet: bool = False,
) -> lgb.LGBMRegressor:
    params     = {**LGBM_BASE, "objective": "quantile", "alpha": alpha}
    spike_mask = y_tr > (floor + tol)
    weights    = np.where(spike_mask, 10.0, 1.0)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    if not quiet:
        log(f"  [{label}]  best_iter={model.best_iteration_}")
    return model


def train_ratio_model(
    X_tr, y_tr, X_val, y_val,
    label: str, quiet: bool = False,
) -> lgb.LGBMRegressor:
    params = {**LGBM_BASE, "objective": "regression"}
    model  = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    if not quiet:
        log(f"  [{label}]  best_iter={model.best_iteration_}")
    return model


def train_spike_classifier(
    X_tr, y_spike_tr, X_val, y_spike_val,
    label: str = "Spike Clf", quiet: bool = False,
) -> Optional[lgb.LGBMClassifier]:
    if int(y_spike_tr.sum()) < 10:
        log(f"  [{label}] skip – spike di train terlalu sedikit ({int(y_spike_tr.sum())})")
        return None

    params = {
        **{k: v for k, v in LGBM_BASE.items()
           if k not in ("early_stopping_rounds",)},
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_spike_tr,
        eval_set=[(X_val, y_spike_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    if not quiet:
        log(f"  [{label}]  best_iter={model.best_iteration_}")
    return model


# ──────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────

def fee_metrics_block(
    y_true: np.ndarray,
    y_pred_med: np.ndarray,
    y_pred_lo: np.ndarray,
    y_pred_hi: np.ndarray,
    floor: float,
    tol: float,
    spike_mask: np.ndarray,
) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred_med))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_med)))
    mask_nf = y_true > (floor + tol)
    mape = (
        float(np.mean(np.abs((y_true[mask_nf] - y_pred_med[mask_nf]) / (y_true[mask_nf] + 1e-10))) * 100)
        if mask_nf.sum() > 0 else 0.0
    )
    ci_cov    = float(np.mean((y_true >= y_pred_lo) & (y_true <= y_pred_hi)) * 100)
    within_10 = float(np.mean(np.abs(y_true - y_pred_med) / (y_true + 1e-10) < 0.10) * 100)

    out = dict(
        mae=mae, rmse=rmse,
        mape_non_floor_pct=mape,
        ci_coverage_p10_p90_pct=ci_cov,
        within_10pct_pct=within_10,
        n=int(len(y_true)),
    )

    sm = np.asarray(spike_mask, dtype=bool)
    if sm.any():
        ys, ps = y_true[sm], y_pred_med[sm]
        out["spike_only"] = {
            "n":    int(sm.sum()),
            "mae":  float(mean_absolute_error(ys, ps)),
            "mape_non_floor_pct": float(np.mean(np.abs((ys - ps) / (ys + 1e-10))) * 100),
        }
    else:
        out["spike_only"] = {"n": 0, "mae": None, "mape_non_floor_pct": None}

    nsm = ~sm
    if nsm.any():
        ys, ps = y_true[nsm], y_pred_med[nsm]
        out["non_spike"] = {"n": int(nsm.sum()), "mae": float(mean_absolute_error(ys, ps))}
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


# ──────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────

def run_training_pipeline(df: pd.DataFrame, floor: float, tol: float) -> dict:
    end_ts = pd.to_datetime(df["interval_ts"]).max()
    train_df, val_df, test_df = temporal_split(df, end_ts, TEST_HOURS, VAL_HOURS)

    if len(train_df) < MIN_TRAIN_ROWS or len(val_df) < 12 or len(test_df) < 12:
        raise RuntimeError(
            f"Data kurang: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    X_tr, y_tr = train_df[FEATURES], train_df["base_fee_gwei"]
    X_va, y_va = val_df[FEATURES],   val_df["base_fee_gwei"]
    X_te, y_te = test_df[FEATURES],  test_df["base_fee_gwei"].values

    log(f"Split (end={end_ts}): train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,}")

    log("\nTraining quantile models...")
    m_lo = train_quantile(X_tr, y_tr, X_va, y_va, 0.10, "Lower  (P10)", floor, tol)
    m_md = train_quantile(X_tr, y_tr, X_va, y_va, 0.50, "Median (P50)", floor, tol)
    m_hi = train_quantile(X_tr, y_tr, X_va, y_va, 0.90, "Upper  (P90)", floor, tol)

    log("Training ratio model...")
    m_ratio = train_ratio_model(
        X_tr, train_df["gas_used_ratio"],
        X_va, val_df["gas_used_ratio"],
        "Ratio",
    )

    log("Training spike classifier...")
    m_spike = None
    if "will_spike_1h" in df.columns:
        m_spike = train_spike_classifier(
            X_tr, train_df["will_spike_1h"],
            X_va, val_df["will_spike_1h"],
        )

    y_plo, y_pmd, y_pup = predict_fee_bundle((m_lo, m_md, m_hi), X_te, floor)
    spike_te = test_df["is_spike"].values

    metrics = fee_metrics_block(y_te, y_pmd, y_plo, y_pup, floor, tol, spike_te)
    metrics["r2_median_oos"] = float(r2_score(y_te, y_pmd))

    # Spike classifier OOS eval
    spike_clf_metrics: dict = {}
    if m_spike is not None and "will_spike_1h" in test_df.columns:
        y_sp_true = test_df["will_spike_1h"].values
        y_sp_prob = m_spike.predict_proba(X_te)[:, 1]
        spike_clf_metrics["spike_base_rate"] = float(y_sp_true.mean())
        spike_clf_metrics["n"] = int(len(y_sp_true))
        if y_sp_true.sum() > 0:
            spike_clf_metrics["auc"] = float(roc_auc_score(y_sp_true, y_sp_prob))

    y_ratio_true = test_df["gas_used_ratio"].values
    metrics["ratio_oos"] = {
        "mae": float(mean_absolute_error(y_ratio_true, m_ratio.predict(X_te))),
        "n":   int(len(y_ratio_true)),
    }

    y_base  = test_df["fee_lag_1"].values
    mae_base = float(mean_absolute_error(y_te, y_base))
    mask_nf  = y_te > (floor + tol)
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
        "end_ts":             end_ts,
        "train_df":           train_df,
        "val_df":             val_df,
        "test_df":            test_df,
        "models":             (m_lo, m_md, m_hi, m_ratio),
        "spike_clf":          m_spike,
        "metrics_primary_oos": metrics,
        "spike_clf_metrics":  spike_clf_metrics,
        "y_test":             y_te,
        "y_pred_median":      y_pmd,
    }


# ──────────────────────────────────────────────
# Walk-forward eval
# ──────────────────────────────────────────────

def walk_forward_eval(df: pd.DataFrame, floor: float, tol: float) -> list:
    ts_max = pd.to_datetime(df["interval_ts"]).max()
    rows   = []
    for w in range(WALK_FORWARD_WINDOWS):
        end_ts = ts_max - timedelta(hours=w * TEST_HOURS)
        train_df, val_df, test_df = temporal_split(df, end_ts, TEST_HOURS, VAL_HOURS)
        if len(train_df) < MIN_TRAIN_ROWS or len(val_df) < 12 or len(test_df) < 12:
            continue
        X_tr, y_tr = train_df[FEATURES], train_df["base_fee_gwei"]
        X_va, y_va = val_df[FEATURES],   val_df["base_fee_gwei"]
        X_te, y_te = test_df[FEATURES],  test_df["base_fee_gwei"].values

        m_md  = train_quantile(X_tr, y_tr, X_va, y_va, 0.50, f"WF P50 w{w}", floor, tol, quiet=True)
        y_pmd = np.clip(m_md.predict(X_te), floor, None)
        mae   = float(mean_absolute_error(y_te, y_pmd))
        mae_l = float(mean_absolute_error(y_te, test_df["fee_lag_1"].values))
        rows.append({
            "window":          w,
            "test_end_ts":     str(end_ts),
            "mae_median_model": mae,
            "mae_baseline_lag1": mae_l,
            "n_test":          len(y_te),
        })
    return rows


# ──────────────────────────────────────────────
# Multi-step horizon evaluation
# ──────────────────────────────────────────────

def _get_hist(lst: list, back: int, default: float) -> float:
    idx = len(lst) - back
    return float(lst[idx]) if idx >= 0 else default


def evaluate_multistep(
    models,
    df: pd.DataFrame,
    floor: float,
    tol: float,
    horizons: list = [12, 36, 72, 144],
) -> dict:
    m_lo, m_md, m_hi, m_ratio = models
    max_h     = max(horizons)
    min_needed = ROLLING_WINDOW + max_h + 24
    if len(df) < min_needed:
        return {}

    horizon_errors: dict = {h: [] for h in horizons}
    step_size = max(24, (len(df) - min_needed) // 4)

    for i in range(4):
        hist_end = ROLLING_WINDOW + i * step_size
        if hist_end + max_h >= len(df):
            break

        history = df.iloc[:hist_end]
        future  = df.iloc[hist_end: hist_end + max_h]
        if len(history) < ROLLING_WINDOW or len(future) < max_h:
            continue

        fee_hist   = list(history["base_fee_gwei"].values)
        ratio_hist = list(history["gas_used_ratio"].values)

        predicted = []
        for j in range(len(future)):
            row_ts = pd.to_datetime(future.iloc[j]["interval_ts"])
            arr    = np.array(fee_hist[-ROLLING_WINDOW:])
            rat_arr = np.array(ratio_hist[-ROLLING_WINDOW:])

            row = {
                "hour":               row_ts.hour,
                "day_of_week":        row_ts.dayofweek,
                "minute":             row_ts.minute,
                "fee_lag_1":          _get_hist(fee_hist, 1, fee_hist[-1]),
                "fee_lag_2":          _get_hist(fee_hist, 2, fee_hist[-1]),
                "fee_lag_3":          _get_hist(fee_hist, 3, fee_hist[-1]),
                "fee_lag_6":          _get_hist(fee_hist, 6, fee_hist[-1]),
                "fee_lag_12":         _get_hist(fee_hist, 12, fee_hist[-1]),
                "fee_lag_24":         _get_hist(fee_hist, 24, fee_hist[-1]),
                "fee_rolling_mean":   float(np.mean(arr)),
                "fee_rolling_std":    float(max(np.std(arr), 1e-5)),
                "fee_rolling_p25":    float(np.percentile(arr, 25)),
                "fee_rolling_p75":    float(np.percentile(arr, 75)),
                "fee_rolling_p90":    float(np.percentile(arr, 90)),
                "fee_diff_1":         _get_hist(fee_hist, 1, fee_hist[-1]) - _get_hist(fee_hist, 2, fee_hist[-1]),
                "fee_diff_2":         _get_hist(fee_hist, 2, fee_hist[-1]) - _get_hist(fee_hist, 3, fee_hist[-1]),
                "fee_above_floor":    max(0.0, _get_hist(fee_hist, 1, fee_hist[-1]) - floor),
                "is_at_floor":        float(_get_hist(fee_hist, 1, fee_hist[-1]) <= floor + tol),
                "fee_zscore":         (_get_hist(fee_hist, 1, fee_hist[-1]) - float(np.mean(arr))) / (float(np.std(arr)) + 1e-10),
                "ratio_lag_1":        _get_hist(ratio_hist, 1, ratio_hist[-1]),
                "ratio_rolling_mean": float(np.mean(rat_arr)),
                "ratio_rolling_std":  float(max(np.std(rat_arr), 1e-5)),
                "ratio_diff_1":       _get_hist(ratio_hist, 1, ratio_hist[-1]) - _get_hist(ratio_hist, 2, ratio_hist[-1]),
                "hour_spike_rate":    0.0,
            }
            X     = pd.DataFrame([row])[FEATURES]
            p_med = float(max(floor, m_md.predict(X)[0]))
            predicted.append(p_med)

            fee_hist.append(p_med)
            next_ratio = float(np.clip(ratio_hist[-1] * 0.97 + 0.03 * 0.5, 0, 1))
            ratio_hist.append(next_ratio)

        predicted  = np.array(predicted)
        actual     = future["base_fee_gwei"].values
        for h in horizons:
            if h <= len(predicted) and h <= len(actual):
                horizon_errors[h].append(float(mean_absolute_error(actual[:h], predicted[:h])))

    return {
        f"mae_h{h*5}min": round(float(np.mean(v)), 8) if v else None
        for h, v in horizon_errors.items()
    }


# ──────────────────────────────────────────────
# Production refit (full data, fixed n_trees)
# ──────────────────────────────────────────────

def refit_production(
    df: pd.DataFrame,
    floor: float,
    tol: float,
    m_lo, m_md, m_hi, m_ratio,
    m_spike=None,
):
    X_full  = df[FEATURES]
    y_fee   = df["base_fee_gwei"]
    y_ratio = df["gas_used_ratio"]
    w_fee   = np.where(y_fee > (floor + tol), 10.0, 1.0)

    def one_q(m, alpha):
        p = {**LGBM_BASE, "objective": "quantile", "alpha": alpha}
        p.pop("early_stopping_rounds", None)
        p["n_estimators"] = effective_n_estimators(m)
        out = lgb.LGBMRegressor(**p)
        out.fit(X_full, y_fee, sample_weight=w_fee)
        return out

    f_lo    = one_q(m_lo, 0.10)
    f_md    = one_q(m_md, 0.50)
    f_hi    = one_q(m_hi, 0.90)

    pr = {**LGBM_BASE, "objective": "regression"}
    pr.pop("early_stopping_rounds", None)
    pr["n_estimators"] = effective_n_estimators(m_ratio)
    f_ratio = lgb.LGBMRegressor(**pr)
    f_ratio.fit(X_full, y_ratio)

    f_spike = None
    has_spike_col = "will_spike_1h" in df.columns and df["will_spike_1h"].sum() >= 10
    if m_spike is not None and has_spike_col:
        bi = getattr(m_spike, "best_iteration_", None)
        n_trees = int(min(500, max(1, bi + 1))) if bi is not None else 500
        pc = {
            **{k: v for k, v in LGBM_BASE.items()
               if k not in ("early_stopping_rounds",)},
            "objective": "binary",
            "metric":    "auc",
            "n_estimators": n_trees,
        }
        f_spike = lgb.LGBMClassifier(**pc)
        f_spike.fit(X_full, df["will_spike_1h"])

    return f_lo, f_md, f_hi, f_ratio, f_spike


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

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
    tol   = float(df_raw["base_fee_gwei"].std() * 0.1)
    log(f"Floor: {floor:.6f} Gwei  tol: {tol:.6f}")

    log("Building hourly profiles...")
    profiles         = build_hourly_profiles(df_raw, floor, tol)
    hour_spike_rates = {h: v["spike_rate"] for h, v in profiles.items()}

    log("Feature engineering...")
    df = build_features(df_raw, floor, tol, hour_spike_rates)
    log(f"Setelah feature engineering: {len(df):,} rows")

    rolling_mean = df["fee_lag_1"].rolling(288, min_periods=12).mean()
    rolling_std  = df["fee_lag_1"].rolling(288, min_periods=12).std()
    df["is_spike"] = ((df["base_fee_gwei"] - rolling_mean) / (rolling_std + 1e-10)) > SPIKE_Z_THRESH

    # Forward-looking spike target: any spike in the next SPIKE_HORIZON intervals
    is_spike_arr = df["is_spike"].astype(np.int8).values
    n             = len(is_spike_arr)
    future_spike  = np.zeros(n, dtype=np.int8)
    for lag in range(1, SPIKE_HORIZON + 1):
        end = n - lag
        if end > 0:
            future_spike[:end] = np.maximum(future_spike[:end], is_spike_arr[lag:n])
    df["will_spike_1h"] = future_spike.astype(int)
    # Trim last SPIKE_HORIZON rows (incomplete future window)
    df = df.iloc[:-SPIKE_HORIZON].reset_index(drop=True)

    spike_rate = df["will_spike_1h"].mean()
    log(f"Spike target rate (will_spike_1h): {spike_rate*100:.2f}%")

    result = run_training_pipeline(df, floor, tol)
    m_lo, m_md, m_hi, m_ratio = result["models"]
    m_spike           = result["spike_clf"]
    metrics           = result["metrics_primary_oos"]
    spike_clf_metrics = result["spike_clf_metrics"]

    log(f"\n{'='*56}")
    log(f"  OOS test ({TEST_HOURS}h, tidak dipakai ES/train)")
    log(f"  MAE (model)   = {metrics['mae']:.8f} Gwei")
    log(f"  MAE (lag-1)   = {metrics['baseline_lag1']['mae']:.8f} Gwei")
    log(f"  Δ MAE         = {metrics['baseline_lag1']['mae_improvement_vs_model']:.8f}")
    log(f"  RMSE          = {metrics['rmse']:.8f} Gwei")
    log(f"  MAPE*         = {metrics['mape_non_floor_pct']:.4f}%  (*non-floor)")
    log(f"  CI P10-P90    = {metrics['ci_coverage_p10_p90_pct']:.1f}%  (nominal ~80%)")
    log(f"  Within 10%    = {metrics['within_10pct_pct']:.1f}%")
    log(f"  R² (median)   = {metrics['r2_median_oos']:.6f}")
    sp = metrics["spike_only"]
    if sp["n"] > 0:
        log(f"  Spike MAE     = {sp['mae']:.8f}  (n={sp['n']})")
    else:
        log("  Spike MAE     = n/a")
    if spike_clf_metrics.get("auc"):
        log(f"  Spike AUC     = {spike_clf_metrics['auc']:.4f}")
    log(f"{'='*56}")

    log("\nWalk-forward evaluation...")
    wf = walk_forward_eval(df, floor, tol)
    for r in wf:
        log(f"  wf{r['window']}: MAE={r['mae_median_model']:.8f}  lag1={r['mae_baseline_lag1']:.8f}  n={r['n_test']}")

    log("\nMulti-step horizon evaluation...")
    multistep = evaluate_multistep((m_lo, m_md, m_hi, m_ratio), df, floor, tol)
    for k, v in multistep.items():
        log(f"  {k}: {v}")

    log("\nRetrain production (full data, n_trees dari ES)...")
    f_lo, f_md, f_hi, f_ratio, f_spike = refit_production(
        df, floor, tol, m_lo, m_md, m_hi, m_ratio, m_spike,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    f_lo.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_lower.txt")
    f_md.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_median.txt")
    f_hi.booster_.save_model(f"{MODEL_DIR}/lgbm_fee_upper.txt")
    f_ratio.booster_.save_model(f"{MODEL_DIR}/lgbm_ratio_median.txt")
    if f_spike is not None:
        f_spike.booster_.save_model(f"{MODEL_DIR}/lgbm_spike_clf.txt")
        log("Spike classifier tersimpan.")
    log("Models tersimpan.")

    # Persist hourly profiles for inference
    profiles_serializable = {str(k): v for k, v in profiles.items()}
    with open(PROFILES_PATH, "w") as fp:
        json.dump(profiles_serializable, fp, indent=2)
    log(f"Hourly profiles disimpan ke {PROFILES_PATH}")

    imp = pd.DataFrame({"feature": FEATURES, "importance": f_md.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    log(f"\nFeature Importances (top 10):\n{imp.head(10).to_string(index=False)}")

    feat_imp_top = [
        {"feature": str(r["feature"]), "importance": int(r["importance"])}
        for _, r in imp.head(14).iterrows()
    ]

    report = {
        "trained_at":    datetime.utcnow().isoformat(),
        "model_version": datetime.utcnow().strftime("v%Y%m%d_%H%M"),
        "architecture":  "LightGBM Quantile Regression + Spike Classifier",
        "evaluation_protocol": {
            "split": f"train | val={VAL_HOURS}h (ES only) | test={TEST_HOURS}h (OOS)",
            "production_refit": "all rows; n_estimators = best_iteration+1 from ES",
            "ci_nominal_central_pct": 80.0,
            "ci_calibration_note":    "Ideal P10-P90 coverage ~80%.",
            "multistep_note":         "Multi-step MAE via recursive 1-step forecast from 4 starting points.",
        },
        "features":           FEATURES,
        "rolling_window":     ROLLING_WINDOW,
        "spike_horizon":      SPIKE_HORIZON,
        "base_l2_floor_gwei": floor,
        "floor_tolerance":    tol,
        "spike_z_threshold":  SPIKE_Z_THRESH,
        "mape_ok":            metrics["mape_non_floor_pct"] < MAPE_THRESHOLD,
        "metrics_oos_test":   metrics,
        "metrics_overall": {
            "mae":         metrics["mae"],
            "rmse":        metrics["rmse"],
            "mape":        metrics["mape_non_floor_pct"],
            "within_10pct": metrics["within_10pct_pct"],
            "r2":          metrics["r2_median_oos"],
        },
        "spike_clf_metrics":       spike_clf_metrics,
        "multistep_mae":           multistep,
        "walk_forward_median_fee": wf,
        "feature_importance_top":  feat_imp_top,
        "ci_coverage_pct":         metrics["ci_coverage_p10_p90_pct"],
        "percentiles": {
            "p1":   float(df["base_fee_gwei"].quantile(0.01)),
            "p25":  float(df["base_fee_gwei"].quantile(0.25)),
            "mean": float(df["base_fee_gwei"].mean()),
            "std":  float(df["base_fee_gwei"].std()),
            "p75":  float(df["base_fee_gwei"].quantile(0.75)),
            "p90":  float(df["base_fee_gwei"].quantile(0.90)),
            "p99":  float(df["base_fee_gwei"].quantile(0.99)),
        },
        "production_trees": {
            "fee_lower":  effective_n_estimators(m_lo),
            "fee_median": effective_n_estimators(m_md),
            "fee_upper":  effective_n_estimators(m_hi),
            "ratio":      effective_n_estimators(m_ratio),
            "spike_clf":  getattr(m_spike, "best_iteration_", None),
        },
        "hourly_profiles": profiles_serializable,
    }

    with open(REPORT_PATH, "w") as fp:
        json.dump(report, fp, indent=2)

    log(f"\nReport disimpan ke {REPORT_PATH}")
    log("Training pipeline selesai.")


if __name__ == "__main__":
    main()
