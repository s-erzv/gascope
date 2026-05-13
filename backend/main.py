import json
import sqlite3
import os
import warnings
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train_model import build_features, FEATURES, ROLLING_WINDOW

warnings.filterwarnings("ignore")

MODEL_DIR    = "model"
DB_PATH      = f"{MODEL_DIR}/gas_fee.db"
REPORT_PATH  = f"{MODEL_DIR}/training_report.json"

WIB_OFFSET       = timedelta(hours=7)
JAKARTA_TZ       = ZoneInfo("Asia/Jakarta")
FORECAST_STEPS   = 289
FREQ             = "5min"
ETH_PRICE_USD    = 3000
MIN_CONFIDENCE   = 0.50
MIN_SAVINGS_PCT  = 1.5

def _load_booster(path: str, required: bool = True) -> Optional[lgb.Booster]:
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"Model tidak ditemukan: {path}")
        return None
    return lgb.Booster(model_file=path)

FEE_LOWER  = _load_booster(f"{MODEL_DIR}/lgbm_fee_lower.txt")
FEE_MEDIAN = _load_booster(f"{MODEL_DIR}/lgbm_fee_median.txt")
FEE_UPPER  = _load_booster(f"{MODEL_DIR}/lgbm_fee_upper.txt")
RATIO_MDL  = _load_booster(f"{MODEL_DIR}/lgbm_ratio_median.txt", required=False)

with open(REPORT_PATH) as f:
    REPORT = json.load(f)

FLOOR_GWEI       = REPORT["base_l2_floor_gwei"]
FLOOR_TOLERANCE  = REPORT["floor_tolerance"]
SPIKE_Z_THRESH   = REPORT["spike_z_threshold"]
GLOBAL_MEAN      = REPORT["percentiles"]["mean"]
GLOBAL_STD       = REPORT["percentiles"]["std"]
GLOBAL_P25       = REPORT["percentiles"]["p25"]
GLOBAL_P75       = REPORT["percentiles"]["p75"]
GLOBAL_P90       = REPORT["percentiles"]["p90"]
GLOBAL_P99       = REPORT["percentiles"]["p99"]
MODEL_VERSION    = REPORT["model_version"]
CI_COVERAGE      = REPORT.get("ci_coverage_pct", 80.0)

app = FastAPI(title="Gascope API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_recent_data(n_rows: int = 600) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT interval_ts, base_fee_gwei, gas_used_ratio
        FROM (
            SELECT * FROM gas_fee
            ORDER BY interval_ts DESC
            LIMIT {n_rows}
        )
        ORDER BY interval_ts ASC
    """, con)
    con.close()

    if df.empty:
        return pd.DataFrame()

    return build_features(df)

def _ts_naive_utc_to_jakarta_epoch_ms(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.tz_convert(JAKARTA_TZ).timestamp() * 1000)

def _ts_naive_utc_to_jakarta_label(ts, *, short: bool = False) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    tj = t.tz_convert(JAKARTA_TZ)
    if short:
        return tj.strftime("%d/%m %H:%M")
    return tj.strftime("%d %b %Y, %H:%M") + " WIB"

def run_forecast(df_history: pd.DataFrame) -> pd.DataFrame:
    now_utc  = datetime.now(timezone.utc).replace(tzinfo=None)
    dates    = pd.date_range(start=now_utc, periods=FORECAST_STEPS, freq=FREQ)

    if df_history.empty or len(df_history) < FORECAST_STEPS:
        return pd.DataFrame({
            "ds":         dates,
            "yhat":       [GLOBAL_MEAN] * FORECAST_STEPS,
            "yhat_lower": [GLOBAL_MEAN * 0.9] * FORECAST_STEPS,
            "yhat_upper": [GLOBAL_MEAN * 1.1] * FORECAST_STEPS,
            "ratio_yhat": [0.5] * FORECAST_STEPS,
        })

    latest = df_history.iloc[-1]
    cur_fee   = float(latest["base_fee_gwei"])
    cur_ratio = float(latest["gas_used_ratio"])

    df_history['dt_hour'] = pd.to_datetime(df_history['interval_ts']).dt.hour
    hourly_fee_profile = df_history.groupby('dt_hour')['base_fee_gwei'].mean().to_dict()
    hourly_ratio_profile = df_history.groupby('dt_hour')['gas_used_ratio'].mean().to_dict()

    frozen_fee_std  = float(max(df_history["base_fee_gwei"].tail(ROLLING_WINDOW).std(), 1e-5))
    frozen_fee_p25  = float(np.percentile(df_history["base_fee_gwei"].tail(ROLLING_WINDOW), 25))
    frozen_fee_p75  = float(np.percentile(df_history["base_fee_gwei"].tail(ROLLING_WINDOW), 75))
    frozen_fee_p90  = float(np.percentile(df_history["base_fee_gwei"].tail(ROLLING_WINDOW), 90))
    frozen_ratio_std  = float(max(df_history["gas_used_ratio"].tail(ROLLING_WINDOW).std(), 1e-5))

    yhat_lower, yhat_median, yhat_upper, ratio_yhat = [], [], [], []
    
    prev_fee_1, prev_fee_2 = cur_fee, cur_fee
    prev_ratio = cur_ratio

    for i in range(FORECAST_STEPS):
        dt = dates[i]
        hr = dt.hour

        target_fee_mean = hourly_fee_profile.get(hr, GLOBAL_MEAN)
        target_ratio_mean = hourly_ratio_profile.get(hr, 0.5)

        row = {
            "hour":               hr,
            "day_of_week":        dt.dayofweek,
            "minute":             dt.minute,
            "fee_lag_1":          cur_fee,
            "fee_lag_2":          prev_fee_1,
            "fee_lag_3":          prev_fee_2,
            "fee_rolling_mean":   target_fee_mean,
            "fee_rolling_std":    frozen_fee_std,
            "fee_rolling_p25":    frozen_fee_p25,
            "fee_rolling_p75":    frozen_fee_p75,
            "fee_rolling_p90":    frozen_fee_p90,
            "fee_diff_1":         cur_fee - prev_fee_1,
            "fee_diff_2":         prev_fee_1 - prev_fee_2,
            "fee_zscore":         (cur_fee - target_fee_mean) / frozen_fee_std,
            "ratio_lag_1":        cur_ratio,
            "ratio_rolling_mean": target_ratio_mean,
            "ratio_rolling_std":  frozen_ratio_std,
            "ratio_diff_1":       cur_ratio - prev_ratio,
        }

        X = pd.DataFrame([row])[FEATURES]

        p_lo  = max(FLOOR_GWEI, float(FEE_LOWER.predict(X)[0]))
        p_med = max(FLOOR_GWEI, float(FEE_MEDIAN.predict(X)[0]))
        p_up  = max(FLOOR_GWEI, float(FEE_UPPER.predict(X)[0]))

        p_lo  = min(p_lo,  p_med)
        p_up  = max(p_up,  p_med)

        if RATIO_MDL:
            next_ratio = float(np.clip(RATIO_MDL.predict(X)[0], 0.0, 1.0))
        else:
            next_ratio = 0.5 + (cur_ratio - 0.5) * 0.97

        expected_mean = p_med + (p_up - p_med) * 0.25

        yhat_lower.append(p_lo)
        yhat_median.append(expected_mean)
        yhat_upper.append(p_up)
        ratio_yhat.append(next_ratio)

        prev_fee_2 = prev_fee_1
        prev_fee_1 = cur_fee
        cur_fee    = (cur_fee * 0.6) + (expected_mean * 0.4)
        
        prev_ratio = cur_ratio
        cur_ratio  = (cur_ratio * 0.6) + (next_ratio * 0.4)

    return pd.DataFrame({
        "ds":         dates,
        "yhat":       yhat_median,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper,
        "ratio_yhat": ratio_yhat,
    })

def z_score(fee: float, mean: float, std: float) -> float:
    return (fee - mean) / (std + 1e-10)

def classify_zone(fee: float, z: float) -> str:
    if fee <= FLOOR_GWEI + FLOOR_TOLERANCE:
        return "FLOOR"
    if z > SPIKE_Z_THRESH:
        return "SPIKE"
    if z > 1.0:
        return "ELEVATED"
    return "NORMAL"

def compute_confidence(fc: pd.DataFrame) -> float:
    ci_widths  = fc["yhat_upper"] - fc["yhat_lower"]
    avg_ci     = float(ci_widths.mean())
    avg_med    = float(fc["yhat"].mean())

    if avg_med < 1e-8:
        return 0.0

    ci_ratio      = avg_ci / avg_med
    ci_factor     = max(0.0, 1.0 - ci_ratio)

    coverage_bonus = (CI_COVERAGE - 70) / 100 if CI_COVERAGE > 70 else 0.0

    return float(min(0.98, max(0.0, ci_factor + coverage_bonus)))

def compute_savings(current_fee: float, fc: pd.DataFrame) -> dict:
    yhat        = fc["yhat"].values
    yhat_lower  = fc["yhat_lower"].values

    min_idx     = int(np.argmin(yhat))
    min_fee     = float(yhat[min_idx])
    min_lower   = float(yhat_lower[min_idx])

    savings_pct       = max(0.0, (current_fee - min_fee)   / (current_fee + 1e-10) * 100)
    savings_pct_cons  = max(0.0, (current_fee - min_lower) / (current_fee + 1e-10) * 100)

    wait_min = min_idx * 5
    wait_hrs = round(wait_min / 60, 1)

    tx_types = [
        ("Transfer ETH",  21_000),
        ("Swap ERC-20",  150_000),
        ("Mint NFT",     200_000),
    ]
    tx_savings = []
    for name, gas in tx_types:
        cost_now  = current_fee * gas / 1e9 * ETH_PRICE_USD
        cost_wait = min_fee     * gas / 1e9 * ETH_PRICE_USD
        saving    = cost_now - cost_wait
        tx_savings.append({
            "tx_type":      name,
            "gas_units":    gas,
            "cost_now_usd":  round(cost_now,  6),
            "cost_wait_usd": round(cost_wait, 6),
            "saving_usd":    round(saving,    6),
            "saving_pct":    round(saving / (cost_now + 1e-10) * 100, 2),
        })

    return {
        "min_fee_gwei":           round(min_fee, 8),
        "min_fee_idx":            min_idx,
        "optimal_wait_minutes":   wait_min,
        "optimal_wait_hours":     wait_hrs,
        "savings_pct":            round(savings_pct, 2),
        "savings_pct_conservative": round(savings_pct_cons, 2),
        "tx_savings":             tx_savings,
    }

def run_rule_engine(
    current_fee:   float,
    current_ratio: float,
    z:             float,
    fc:            pd.DataFrame,
    confidence:    float,
    savings:       dict,
) -> dict:
    
    zone        = classify_zone(current_fee, z)
    at_floor    = zone == "FLOOR"
    savings_pct = savings["savings_pct"]
    wait_hrs    = savings["optimal_wait_hours"]
    wait_min    = savings["optimal_wait_minutes"]

    ratio_1h    = fc["ratio_yhat"].iloc[:12].values
    ratio_trend = float(ratio_1h[-1] - ratio_1h[0])
    ratio_1h_avg = float(ratio_1h.mean())

    fee_1h_avg  = float(fc["yhat"].iloc[:12].mean())
    fee_trend   = fee_1h_avg - current_fee

    fee_24h_min = float(fc["yhat"].min())
    min_at_floor = fee_24h_min <= (FLOOR_GWEI + FLOOR_TOLERANCE * 3)

    base = dict(
        zone=zone,
        z_score=round(z, 3),
        confidence_score=round(confidence, 3),
        current_fee_gwei=round(current_fee, 8),
        lowest_future_gwei=savings["min_fee_gwei"],
        savings_estimate_pct=savings_pct,
        savings_pct_conservative=savings["savings_pct_conservative"],
        optimal_wait_hours=wait_hrs,
        optimal_wait_minutes=wait_min,
        fee_1h_trend=round(fee_trend, 8),
        ratio_1h_avg=round(ratio_1h_avg, 4),
        ratio_increasing=ratio_trend > 0.10,
        urgency_score=round(min(1.0, max(0.0,
            (z / (SPIKE_Z_THRESH + 1e-10)) * 0.4 +
            (current_ratio / 1.0) * 0.3 +
            (abs(fee_trend) / (GLOBAL_STD + 1e-10)) * 0.3
        )), 3),
        rule_triggered="",
    )

    if zone == "SPIKE" and confidence >= MIN_CONFIDENCE:
        return {**base,
            "action":         "SPIKE_ALERT",
            "rule_triggered": "active_spike",
            "message": (
                f" Spike aktif (z={z:.1f}σ). Fee sedang {current_fee:.4f} Gwei, "
                f"prediksi turun ke {savings['min_fee_gwei']:.4f} Gwei "
                f"(hemat {savings_pct:.1f}%) dalam ~{wait_hrs} jam. "
                f"Tunda transaksi sekarang."
            ),
        }

    if (not at_floor
            and ratio_trend > 0.12
            and current_ratio > 0.75
            and confidence >= MIN_CONFIDENCE):
        return {**base,
            "action":         "SPIKE_ALERT",
            "rule_triggered": "congestion_rising",
            "message": (
                f" Network congestion naik cepat (ratio {current_ratio:.2f} → "
                f"trend +{ratio_trend:.2f}/jam). Fee kemungkinan spike. "
                f"Eksekusi SEKARANG atau tunggu sampai spike reda."
            ),
        }

    if (at_floor
            and current_ratio < 0.65
            and fee_trend <= (GLOBAL_STD * 0.3)
            and confidence >= MIN_CONFIDENCE):
        return {**base,
            "action":         "EXECUTE_NOW",
            "rule_triggered": "floor_stable",
            "message": (
                f" Kondisi optimal. Fee di minimum ({current_fee:.4f} Gwei), "
                f"network longgar (ratio {current_ratio:.2f}). "
                f"Waktu terbaik untuk transaksi sekarang."
            ),
        }

    if (not at_floor
            and min_at_floor
            and wait_hrs <= 3.0
            and savings_pct > MIN_SAVINGS_PCT
            and confidence >= MIN_CONFIDENCE):
        return {**base,
            "action":         "EXECUTE_SOON",
            "rule_triggered": "fee_returning_floor",
            "message": (
                f" Fee akan turun ke minimum dalam ~{wait_min} menit. "
                f"Potensi hemat {savings_pct:.1f}%. "
                f"Tunggu sebentar lagi untuk transaksi optimal."
            ),
        }

    if (zone in ("ELEVATED", "SPIKE")
            and current_ratio > 0.72
            and savings_pct > MIN_SAVINGS_PCT
            and confidence >= MIN_CONFIDENCE):
        return {**base,
            "action":         "HOLD",
            "rule_triggered": "elevated_congested",
            "message": (
                f"⏸ Fee {zone.lower()} + network padat (ratio {current_ratio:.2f}). "
                f"Tunggu ~{wait_hrs} jam untuk hemat {savings_pct:.1f}% "
                f"(prediksi min {savings['min_fee_gwei']:.4f} Gwei)."
            ),
        }

    extra = ""
    if savings_pct > MIN_SAVINGS_PCT:
        extra = f" Potensi hemat {savings_pct:.1f}% jika tunggu {wait_hrs} jam."
    msg = (
        f"Fee normal ({current_fee:.4f} Gwei, z={z:.2f}σ). "
        f"Tidak ada anomali kuat.{extra}"
    )
    if at_floor:
        msg = f"Fee di minimum ({current_fee:.4f} Gwei). Aman untuk transaksi.{extra}"

    return {**base,
        "action":         "MONITOR",
        "rule_triggered": "default_monitor",
        "message":        msg,
    }

class ChartPoint(BaseModel):
    time_epoch_ms:   int
    time_label:      str
    segment:         Literal["historical", "forecast"] = "forecast"
    actual_fee_gwei: Optional[float] = None
    fee_gwei:        Optional[float] = None
    lower_gwei:      Optional[float] = None
    upper_gwei:      Optional[float] = None
    ratio_forecast:  Optional[float] = None

class TxCost(BaseModel):
    tx_type:      str
    gas_units:    int
    cost_now_usd:  float
    cost_wait_usd: float
    saving_usd:    float
    saving_pct:    float

class Recommendation(BaseModel):
    action:           Literal["EXECUTE_NOW", "EXECUTE_SOON", "SPIKE_ALERT", "HOLD", "MONITOR"]
    zone:             Literal["FLOOR", "NORMAL", "ELEVATED", "SPIKE"]
    message:          str
    confidence_score: float
    z_score:          float
    urgency_score:    float
    current_fee_gwei: float
    lowest_future_gwei: float
    savings_estimate_pct: float
    savings_pct_conservative: float
    optimal_wait_hours:   Optional[float]
    optimal_wait_minutes: Optional[int]
    fee_1h_trend:     float
    ratio_1h_avg:     Optional[float]
    ratio_increasing: bool
    rule_triggered:   str

class PredictResponse(BaseModel):
    timestamp_utc:    str
    timestamp_wib:    str
    model_version:    str
    data_quality:     Literal["fresh", "stale"]
    data_source:      str
    latest_data_ts_wib: Optional[str]
    recommendation:   Recommendation
    chart_data:       list[ChartPoint]
    tx_costs:         list[TxCost]
    percentiles:      dict
    chain_stats:      dict
    chart_timezone:   str = "Asia/Jakarta"
    chart_range_wib:  Optional[str] = None

@app.get("/")
def root():
    return {"service": "Gascope API", "version": MODEL_VERSION, "status": "ok"}

@app.get("/api/predict", response_model=PredictResponse)
def get_prediction():
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    df_hist = load_recent_data(600)

    if df_hist.empty:
        data_quality   = "stale"
        current_fee    = GLOBAL_MEAN
        current_ratio  = 0.5
        ts_label       = None
        roll_mean      = GLOBAL_MEAN
        roll_std       = GLOBAL_STD
    else:
        latest         = df_hist.iloc[-1]
        current_fee    = float(latest["base_fee_gwei"])
        current_ratio  = float(latest["gas_used_ratio"])
        roll_mean      = float(latest["fee_rolling_mean"])
        roll_std       = float(latest["fee_rolling_std"])
        ts_label       = _ts_naive_utc_to_jakarta_label(latest["interval_ts"], short=False)

        last_ts = pd.to_datetime(latest["interval_ts"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
        data_quality = "fresh" if age_min <= 15 else "stale"

    fc = run_forecast(df_hist)

    z          = z_score(current_fee, roll_mean, roll_std)
    confidence = compute_confidence(fc)
    savings    = compute_savings(current_fee, fc)

    rec = run_rule_engine(
        current_fee=current_fee,
        current_ratio=current_ratio,
        z=z,
        fc=fc,
        confidence=confidence,
        savings=savings,
    )

    chart_data: list[ChartPoint] = []

    if not df_hist.empty:
        past = df_hist.tail(144)
        X_hist = past[FEATURES]
        preds_hist = FEE_MEDIAN.predict(X_hist)
        
        for i, (idx, row) in enumerate(past.iterrows()):
            ts_raw = row["interval_ts"]
            ems = _ts_naive_utc_to_jakarta_epoch_ms(ts_raw)
            chart_data.append(ChartPoint(
                time_epoch_ms=ems,
                time_label=_ts_naive_utc_to_jakarta_label(ts_raw, short=False),
                segment="historical",
                actual_fee_gwei=round(float(row["base_fee_gwei"]), 8),
                fee_gwei=round(max(FLOOR_GWEI, float(preds_hist[i])), 8),
            ))

    for _, row in fc.iterrows():
        ts_raw = row["ds"]
        ems = _ts_naive_utc_to_jakarta_epoch_ms(ts_raw)
        chart_data.append(ChartPoint(
            time_epoch_ms=ems,
            time_label=_ts_naive_utc_to_jakarta_label(ts_raw, short=False),
            segment="forecast",
            fee_gwei=round(float(row["yhat"]), 8),
            lower_gwei=round(float(row["yhat_lower"]), 8),
            upper_gwei=round(float(row["yhat_upper"]), 8),
            ratio_forecast=round(float(row["ratio_yhat"]), 4),
        ))

    chart_data.sort(key=lambda c: c.time_epoch_ms)
    chart_range_wib = None
    if len(chart_data) >= 2:
        chart_range_wib = f"{chart_data[0].time_label} → {chart_data[-1].time_label}"

    return PredictResponse(
        timestamp_utc=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp_wib=(now_utc + WIB_OFFSET).strftime("%Y-%m-%d %H:%M:%S"),
        model_version=MODEL_VERSION,
        data_quality=data_quality,
        data_source="Base L2 RPC · LightGBM v3",
        latest_data_ts_wib=ts_label,
        recommendation=Recommendation(**rec),
        chart_data=chart_data,
        tx_costs=[TxCost(**t) for t in savings["tx_savings"]],
        percentiles={
            "p25":   round(GLOBAL_P25, 8),
            "p75":   round(GLOBAL_P75, 8),
            "p90":   round(GLOBAL_P90, 8),
            "p99":   round(GLOBAL_P99, 8),
            "floor": FLOOR_GWEI,
        },
        chain_stats={
            "floor_gwei":    FLOOR_GWEI,
            "global_mean":   round(GLOBAL_MEAN, 8),
            "global_std":    round(GLOBAL_STD, 8),
            "z_score":       round(z, 3),
            "current_ratio": round(current_ratio, 4),
            "roll_mean_6h":  round(roll_mean, 8),
            "roll_std_6h":   round(roll_std, 8),
        },
        chart_timezone="Asia/Jakarta",
        chart_range_wib=chart_range_wib,
    )

@app.get("/api/metrics")
def get_metrics():
    return {
        "model_version":   MODEL_VERSION,
        "architecture":    REPORT.get("architecture"),
        "features":        FEATURES,
        "trained_at":      REPORT.get("trained_at"),
        "metrics_overall": REPORT.get("metrics_overall"),
        "ci_coverage_pct": REPORT.get("ci_coverage_pct"),
        "mape_ok":         REPORT.get("mape_ok"),
        "percentiles":     REPORT.get("percentiles"),
        "evaluation_protocol": REPORT.get("evaluation_protocol"),
        "metrics_oos_test":    REPORT.get("metrics_oos_test"),
        "walk_forward_median_fee": REPORT.get("walk_forward_median_fee"),
        "feature_importance_top":  REPORT.get("feature_importance_top"),
    }

@app.get("/api/status")
def get_status():
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT MAX(interval_ts), COUNT(*) FROM gas_fee").fetchone()
    con.close()

    last_ts_str, row_count = row
    data_age_hours = None
    if last_ts_str:
        last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        data_age_hours = round((datetime.now(timezone.utc) - last_ts).total_seconds() / 3600, 2)

    mape = REPORT.get("metrics_overall", {}).get("mape", 999)
    return {
        "status":           "ok",
        "model_version":    MODEL_VERSION,
        "row_count":        row_count,
        "data_age_hours":   data_age_hours,
        "data_stale":       data_age_hours is not None and data_age_hours > 0.25,
        "model_degraded":   mape > 25.0,
        "mape":             round(mape, 2),
        "ci_coverage_pct":  round(REPORT.get("ci_coverage_pct", 0), 1),
        "floor_gwei":       FLOOR_GWEI,
        "server_time_wib":  (datetime.utcnow() + WIB_OFFSET).strftime("%Y-%m-%d %H:%M:%S"),
    }