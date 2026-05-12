import json
import os
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
from pydantic import BaseModel
import warnings

warnings.filterwarnings("ignore")

FEE_MODEL_LOWER_PATH  = "model/lgbm_fee_lower.txt"
FEE_MODEL_MEDIAN_PATH = "model/lgbm_fee_median.txt"
FEE_MODEL_UPPER_PATH  = "model/lgbm_fee_upper.txt"
RATIO_MODEL_PATH      = "model/lgbm_ratio_median.txt"
REPORT_PATH           = "model/training_report.json"
DATA_PATH             = "model/gas_fee_historical.csv"

WIB_OFFSET          = timedelta(hours=7)
FORECAST_PERIODS    = 289
FREQ                = "5min"
ETH_PRICE_USD       = 3000

BASE_L2_FLOOR_GWEI  = 10.0
FLOOR_TOLERANCE     = 2.0
SPIKE_Z_THRESHOLD   = 2.0

RATIO_CONGESTION_THRESHOLD   = 0.80
RATIO_HIGH_THRESHOLD         = 0.90
SPIKE_IMMINENT_RATIO_DELTA   = 0.15
FEE_RECOVERY_THRESHOLD       = 0.85
MIN_SAVINGS_PCT              = 2.0
MIN_CONFIDENCE               = 0.55
MAPE_DEGRADED_THRESHOLD      = 25.0

app = FastAPI(title="Gascope API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    FEE_MODEL_LOWER  = lgb.Booster(model_file=FEE_MODEL_LOWER_PATH)
    FEE_MODEL_MEDIAN = lgb.Booster(model_file=FEE_MODEL_MEDIAN_PATH)
    FEE_MODEL_UPPER  = lgb.Booster(model_file=FEE_MODEL_UPPER_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model Fee LightGBM: {e}")

RATIO_MODEL = None
try:
    RATIO_MODEL = lgb.Booster(model_file=RATIO_MODEL_PATH)
except Exception:
    pass

try:
    with open(REPORT_PATH, "r") as f:
        REPORT = json.load(f)

    BASE_L2_FLOOR_GWEI  = REPORT.get("base_l2_floor_gwei", BASE_L2_FLOOR_GWEI)
    FLOOR_TOLERANCE     = REPORT.get("floor_tolerance",    FLOOR_TOLERANCE)
    SPIKE_Z_THRESHOLD   = REPORT.get("spike_z_threshold",  SPIKE_Z_THRESHOLD)

    GLOBAL_P25    = REPORT["percentiles"].get("p25", 0.0)
    GLOBAL_P75    = REPORT["percentiles"]["p75"]
    GLOBAL_P90    = REPORT["percentiles"]["p90"]
    GLOBAL_P99    = REPORT["percentiles"]["p99"]
    GLOBAL_MEAN   = REPORT["percentiles"]["mean"]
    GLOBAL_STD    = REPORT["percentiles"]["std"]
    
    MODEL_VERSION = REPORT.get("model_version", "unknown")
    MODEL_FEATURES = REPORT.get("features", [])
    HAS_RATIO_MODEL = RATIO_MODEL is not None

except FileNotFoundError:
    raise RuntimeError(f"Report tidak ditemukan: {REPORT_PATH}")

def get_current_data() -> tuple[Optional[float], float, Optional[str], dict]:
    try:
        cols = ["timestamp", "base_fee_gwei"]
        if "gas_used_ratio" in MODEL_FEATURES or "ratio_lag_1" in MODEL_FEATURES:
            cols.append("gas_used_ratio")

        df = pd.read_csv(DATA_PATH, usecols=cols).tail(300)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

        window = 72
        df["fee_rolling_mean"] = df["base_fee_gwei"].rolling(window, min_periods=6).mean()
        df["fee_rolling_std"]  = df["base_fee_gwei"].rolling(window, min_periods=6).std()
        df["fee_rolling_p75"]  = df["base_fee_gwei"].rolling(window, min_periods=6).quantile(0.75)

        if "gas_used_ratio" in df.columns:
            df["ratio_rolling_mean"] = df["gas_used_ratio"].rolling(window, min_periods=6).mean()
            df["ratio_rolling_std"]  = df["gas_used_ratio"].rolling(window, min_periods=6).std()

        latest = df.iloc[-1]
        fee    = float(latest["base_fee_gwei"])
        ratio  = float(latest["gas_used_ratio"]) if "gas_used_ratio" in df.columns else 0.5
        ts_wib = (latest["timestamp"] + WIB_OFFSET).strftime("%d %b %Y, %H:%M WIB")

        latest_features = {
            "fee_lag_1": fee,
            "fee_rolling_mean": float(latest["fee_rolling_mean"]) if not pd.isna(latest["fee_rolling_mean"]) else GLOBAL_MEAN,
            "fee_rolling_std": float(latest["fee_rolling_std"]) if not pd.isna(latest["fee_rolling_std"]) else GLOBAL_STD,
            "fee_rolling_p75": float(latest["fee_rolling_p75"]) if not pd.isna(latest["fee_rolling_p75"]) else GLOBAL_P75,
        }
        
        if "gas_used_ratio" in df.columns:
            latest_features["ratio_lag_1"] = ratio
            latest_features["gas_used_ratio"] = ratio
            latest_features["ratio_rolling_mean"] = float(latest["ratio_rolling_mean"]) if not pd.isna(latest["ratio_rolling_mean"]) else 0.5
            latest_features["ratio_rolling_std"] = float(latest["ratio_rolling_std"]) if not pd.isna(latest["ratio_rolling_std"]) else 0.05

        return fee, ratio, ts_wib, latest_features

    except Exception as e:
        print(f"[WARNING] get_current_data error: {e}")
        return None, 0.5, None, {}

def build_future_df(now_utc: datetime, latest_features: dict) -> pd.DataFrame:
    dates = pd.date_range(start=now_utc, periods=FORECAST_PERIODS, freq=FREQ)
    df = pd.DataFrame({"ds": dates})
    
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    
    for feat, val in latest_features.items():
        df[feat] = val
        
    return df

def forecast_fee(now_utc: datetime, latest_features: dict) -> pd.DataFrame:
    df = build_future_df(now_utc, latest_features)
    
    X_pred = df[MODEL_FEATURES]
    
    yhat_lower = FEE_MODEL_LOWER.predict(X_pred)
    yhat_median = FEE_MODEL_MEDIAN.predict(X_pred)
    yhat_upper = FEE_MODEL_UPPER.predict(X_pred)
    
    df["yhat_lower"] = np.clip(yhat_lower, BASE_L2_FLOOR_GWEI, None)
    df["yhat"]       = np.clip(yhat_median, BASE_L2_FLOOR_GWEI, None)
    df["yhat_upper"] = np.clip(yhat_upper, BASE_L2_FLOOR_GWEI, None)
    return df

def forecast_ratio(now_utc: datetime, latest_features: dict) -> Optional[pd.DataFrame]:
    if not HAS_RATIO_MODEL:
        return None
        
    df = build_future_df(now_utc, latest_features)
    X_pred = df[MODEL_FEATURES]
    
    yhat = RATIO_MODEL.predict(X_pred)
    df["yhat"] = np.clip(yhat, 0, 1)
    return df


def calculate_confidence(fc_fee: pd.DataFrame) -> float:
    ci_widths = fc_fee["yhat_upper"] - fc_fee["yhat_lower"]
    avg_ci    = float(ci_widths.mean())
    avg_pred  = float(fc_fee["yhat"].mean())
    if avg_pred <= BASE_L2_FLOOR_GWEI:
        return 0.85
    ci_ratio = avg_ci / avg_pred
    return float(max(0.0, min(1.0, 1.0 - (ci_ratio / 2))))

def compute_fee_zscore(current_fee: float, roll_mean: float, roll_std: float) -> float:
    if roll_std < 1e-10:
        return 0.0
    return (current_fee - roll_mean) / roll_std

def is_at_floor(fee: float) -> bool:
    return fee <= (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE)

def classify_zone(current_fee: float, z_score: float) -> str:
    if is_at_floor(current_fee):
        return "FLOOR"
    if z_score > SPIKE_Z_THRESHOLD:
        return "SPIKE"
    if z_score > 1.0:
        return "ELEVATED"
    return "NORMAL"

def compute_savings_window(current_fee: float, fc_fee: pd.DataFrame) -> dict:
    yhat = fc_fee["yhat"].values
    yhat_lower = fc_fee["yhat_lower"].values

    min_pred    = float(yhat.min())
    min_idx     = int(yhat.argmin())
    min_lower   = float(yhat_lower[min_idx])

    savings_vs_current = max(0, (current_fee - min_pred) / current_fee * 100)
    savings_conservative = max(0, (current_fee - min_lower) / current_fee * 100)

    wait_minutes  = min_idx * 5
    wait_hours    = round(wait_minutes / 60, 1)

    tx_types = [
        ("Swap ERC-20",  150_000),
        ("Mint NFT",     200_000),
        ("Transfer ETH",  21_000),
    ]
    tx_savings = []
    for name, gas in tx_types:
        cost_now  = current_fee * gas * (1 / 1e9) * ETH_PRICE_USD
        cost_opt  = min_pred    * gas * (1 / 1e9) * ETH_PRICE_USD
        saving    = cost_now - cost_opt
        tx_savings.append({
            "tx_type": name, "gas_units": gas,
            "cost_now_usd":  round(cost_now, 6),
            "cost_wait_usd": round(cost_opt, 6),
            "saving_usd":    round(saving, 6),
            "saving_pct":    round(saving / cost_now * 100, 2) if cost_now > 0 else 0.0,
        })

    return {
        "min_fee_gwei":            round(min_pred, 8),
        "min_fee_idx":             min_idx,
        "optimal_wait_minutes":    wait_minutes,
        "optimal_wait_hours":      wait_hours,
        "savings_pct":             round(savings_vs_current, 2),
        "savings_pct_conservative": round(savings_conservative, 2),
        "tx_savings":              tx_savings,
    }

def run_rule_engine(
    current_fee:  float,
    fc_fee:       pd.DataFrame,
    fc_ratio:     Optional[pd.DataFrame],
    current_ratio: float,
    z_score:      float,
    confidence:   float,
    savings:      dict,
    roll_std:     float
) -> dict:
    at_floor    = is_at_floor(current_fee)
    zone        = classify_zone(current_fee, z_score)
    savings_pct = savings["savings_pct"]

    fee_1h_avg   = float(fc_fee["yhat"].iloc[:12].mean())
    fee_24h_min  = float(fc_fee["yhat"].min())
    fee_1h_trend = fee_1h_avg - current_fee

    ratio_1h_avg   = None
    ratio_increasing = False
    if fc_ratio is not None:
        ratio_1h = fc_ratio["yhat"].iloc[:12].values
        ratio_1h_avg = float(ratio_1h.mean())
        ratio_increasing = float(ratio_1h[-1] - ratio_1h[0]) > SPIKE_IMMINENT_RATIO_DELTA

    urgency = min(1.0, max(0.0,
        (z_score / SPIKE_Z_THRESHOLD) * 0.4
        + (current_ratio / 1.0) * 0.3
        + (abs(fee_1h_trend) / (roll_std + 1e-10)) * 0.3
    ))

    base_result = {
        "zone":              zone,
        "z_score":           round(z_score, 3),
        "confidence_score":  round(confidence, 3),
        "current_fee_gwei":  round(current_fee, 8),
        "lowest_future_gwei": savings["min_fee_gwei"],
        "savings_estimate_pct": savings_pct if savings_pct > MIN_SAVINGS_PCT else 0.0,
        "savings_pct_conservative": savings["savings_pct_conservative"],
        "optimal_wait_hours": savings["optimal_wait_hours"] if savings_pct > MIN_SAVINGS_PCT else None,
        "optimal_wait_minutes": savings["optimal_wait_minutes"] if savings_pct > MIN_SAVINGS_PCT else None,
        "urgency_score":     round(urgency, 3),
        "fee_1h_trend":      round(fee_1h_trend, 8),
        "ratio_1h_avg":      round(ratio_1h_avg, 4) if ratio_1h_avg is not None else None,
        "ratio_increasing":  ratio_increasing,
    }

    if zone == "SPIKE" and confidence >= MIN_CONFIDENCE:
        recovery_pct = savings_pct
        wait_h = savings["optimal_wait_hours"]
        return {
            **base_result,
            "action": "SPIKE_ALERT",
            "rule_triggered": "active_spike_detected",
            "message": (
                f"Gas fee spike aktif (z={z_score:.1f}σ di atas normal). "
                f"Prediksi turun {recovery_pct:.1f}% dalam ~{wait_h} jam. "
                f"Tunda transaksi untuk hemat biaya."
            ),
        }

    if (not at_floor and ratio_increasing and
            current_ratio > RATIO_CONGESTION_THRESHOLD and
            confidence >= MIN_CONFIDENCE):
        return {
            **base_result,
            "action": "SPIKE_ALERT",
            "rule_triggered": "spike_imminent_ratio_rising",
            "message": (
                f"Network congestion meningkat (ratio={current_ratio:.2f}, naik cepat). "
                f"Fee kemungkinan akan naik dalam 1 jam ke depan. "
                f"Eksekusi SEKARANG sebelum spike, atau tunggu spike selesai."
            ),
        }

    if (at_floor and
            current_ratio < 0.7 and
            fee_1h_trend >= 0 and
            confidence >= MIN_CONFIDENCE):
        return {
            **base_result,
            "action": "EXECUTE_NOW",
            "rule_triggered": "floor_price_stable_network",
            "message": (
                f"Fee di minimum jaringan ({current_fee:.6f} Gwei) and network longgar "
                f"(ratio={current_ratio:.2f}). Ini waktu optimal untuk transaksi."
            ),
        }

    fee_returns_to_floor = fee_24h_min <= (BASE_L2_FLOOR_GWEI + FLOOR_TOLERANCE * 2)
    if (zone in ("ELEVATED", "NORMAL") and
            fee_returns_to_floor and
            savings["optimal_wait_hours"] <= 3.0 and
            savings_pct > MIN_SAVINGS_PCT):
        wait_m = savings["optimal_wait_minutes"]
        return {
            **base_result,
            "action": "EXECUTE_SOON",
            "rule_triggered": "fee_returning_to_floor",
            "message": (
                f"🕐 Fee akan kembali ke minimum dalam ~{wait_m} menit. "
                f"Potensi hemat {savings_pct:.1f}%. Tunggu sebentar lagi."
            ),
        }

    if (zone in ("ELEVATED", "SPIKE") and
            current_ratio > RATIO_CONGESTION_THRESHOLD and
            savings_pct > MIN_SAVINGS_PCT and
            confidence >= MIN_CONFIDENCE):
        wait_h = savings["optimal_wait_hours"]
        return {
            **base_result,
            "action": "HOLD",
            "rule_triggered": "elevated_fee_congested_network",
            "message": (
                f"Fee elevated + network padat (ratio={current_ratio:.2f}). "
                f"Tunda transaksi ~{wait_h} jam untuk hemat {savings_pct:.1f}%."
            ),
        }

    extra = ""
    if savings_pct > MIN_SAVINGS_PCT:
        extra = f" Potensi saving {savings_pct:.1f}% kalau tunggu {savings['optimal_wait_hours']} jam."
    if at_floor:
        msg = f"Fee di minimum jaringan. Aman untuk transaksi kapan saja.{extra}"
    else:
        msg = f"Kondisi normal. Tidak ada sinyal kuat.{extra}"

    return {
        **base_result,
        "action": "MONITOR",
        "rule_triggered": "default_monitor",
        "message": msg,
    }

class ChartPoint(BaseModel):
    time_epoch_ms: int
    time_label: str
    fee_gwei: float
    actual_fee_gwei: Optional[float] = None
    lower_gwei: float
    upper_gwei: float
    ratio_forecast: Optional[float] = None

class TxCost(BaseModel):
    tx_type: str
    gas_units: int
    cost_now_usd: float
    cost_wait_usd: float
    saving_usd: float
    saving_pct: float

class Recommendation(BaseModel):
    action: Literal["EXECUTE_NOW", "EXECUTE_SOON", "SPIKE_ALERT", "HOLD", "MONITOR"]
    zone: Literal["FLOOR", "NORMAL", "ELEVATED", "SPIKE"]
    message: str
    confidence_score: float
    z_score: float
    urgency_score: float
    current_fee_gwei: float
    lowest_future_gwei: float
    savings_estimate_pct: float
    savings_pct_conservative: float
    optimal_wait_hours: Optional[float]
    optimal_wait_minutes: Optional[int]
    fee_1h_trend: float
    ratio_1h_avg: Optional[float]
    ratio_increasing: bool
    rule_triggered: str

class PredictResponse(BaseModel):
    timestamp_utc: str
    timestamp_wib: str
    model_version: str
    data_quality: Literal["fresh", "stale"]
    data_source: str
    latest_data_ts_wib: Optional[str]
    recommendation: Recommendation
    chart_data: list[ChartPoint]
    tx_costs: list[TxCost]
    percentiles: dict
    chain_stats: dict

# ENDPOINTSZZZZZZ
@app.get("/")
def root():
    return {
        "service": "Gascope API",
        "version": MODEL_VERSION,
        "chain": "Base L2",
        "mode": "LightGBM Quantile Prescriptive",
        "status": "running",
    }

@app.get("/api/predict", response_model=PredictResponse)
def get_prediction():
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    now_wib = now_utc + WIB_OFFSET

    start_of_day_wib = now_wib.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_day_utc = start_of_day_wib - WIB_OFFSET

    current_fee, gas_ratio, ts_label, latest_features = get_current_data()
    data_quality = "fresh"
    
    roll_mean = latest_features.get("fee_rolling_mean", GLOBAL_MEAN)
    roll_std  = latest_features.get("fee_rolling_std", GLOBAL_STD)

    fc_fee_full = forecast_fee(start_of_day_utc, latest_features)
    fc_ratio_full = forecast_ratio(start_of_day_utc, latest_features)

    if current_fee is None:
        idx_now = min(range(len(fc_fee_full)), key=lambda i: abs(fc_fee_full.iloc[i]["ds"] - now_utc))
        current_fee  = float(fc_fee_full.iloc[idx_now]["yhat"])
        data_quality = "stale"

    fc_fee_future = fc_fee_full[fc_fee_full["ds"] >= now_utc].copy()
    if fc_fee_future.empty:
        fc_fee_future = fc_fee_full.tail(12)

    fc_ratio_future = None
    if fc_ratio_full is not None:
        fc_ratio_future = fc_ratio_full[fc_ratio_full["ds"] >= now_utc].copy()

    z_score    = compute_fee_zscore(current_fee, roll_mean, roll_std)
    confidence = calculate_confidence(fc_fee_future)
    savings    = compute_savings_window(current_fee, fc_fee_future)

    rec = run_rule_engine(
        current_fee=current_fee,
        fc_fee=fc_fee_future,
        fc_ratio=fc_ratio_future,
        current_ratio=gas_ratio or 0.5,
        z_score=z_score,
        confidence=confidence,
        savings=savings,
        roll_std=roll_std
    )

    actual_today = pd.DataFrame()
    try:
        if os.path.exists(DATA_PATH):
            df_hist = pd.read_csv(DATA_PATH, usecols=["timestamp", "base_fee_gwei"])
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], utc=True).dt.tz_localize(None)
            actual_today = df_hist[df_hist["timestamp"] >= start_of_day_utc].copy()
    except Exception as e:
        print(f"Error loading historical data: {e}")

    chart_data = []
    for i, (_, row) in enumerate(fc_fee_full.iterrows()):
        dt_wib   = row["ds"] + WIB_OFFSET
        epoch_ms = int(dt_wib.timestamp() * 1000)

        actual_val = None
        if not actual_today.empty and row["ds"] <= now_utc + timedelta(minutes=2):
            mask = (actual_today["timestamp"] >= row["ds"] - timedelta(minutes=3)) & \
                   (actual_today["timestamp"] <= row["ds"] + timedelta(minutes=3))
            if not actual_today[mask].empty:
                actual_val = float(actual_today[mask].iloc[-1]["base_fee_gwei"])

        ratio_fc = float(fc_ratio_full.iloc[i]["yhat"]) if fc_ratio_full is not None else None
        chart_data.append(ChartPoint(
            time_epoch_ms   = epoch_ms,
            time_label      = dt_wib.strftime("%H:%M"),
            fee_gwei        = round(float(row["yhat"]), 8),
            actual_fee_gwei = round(actual_val, 8) if actual_val is not None else None,
            lower_gwei      = round(float(row["yhat_lower"]), 8),
            upper_gwei      = round(float(row["yhat_upper"]), 8),
            ratio_forecast  = round(ratio_fc, 4) if ratio_fc is not None else None,
        ))

    return PredictResponse(
        timestamp_utc       = now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp_wib       = (now_utc + WIB_OFFSET).strftime("%Y-%m-%d %H:%M:%S"),
        model_version       = MODEL_VERSION,
        data_quality        = data_quality,
        data_source         = "Base L2 RPC · LightGBM Engine",
        latest_data_ts_wib  = ts_label,
        recommendation      = Recommendation(
            action                   = rec["action"],
            zone                     = rec["zone"],
            message                  = rec["message"],
            confidence_score         = rec["confidence_score"],
            z_score                  = rec["z_score"],
            urgency_score            = rec["urgency_score"],
            current_fee_gwei         = rec["current_fee_gwei"],
            lowest_future_gwei       = rec["lowest_future_gwei"],
            savings_estimate_pct     = rec["savings_estimate_pct"],
            savings_pct_conservative = rec["savings_pct_conservative"],
            optimal_wait_hours       = rec["optimal_wait_hours"],
            optimal_wait_minutes     = rec["optimal_wait_minutes"],
            fee_1h_trend             = rec["fee_1h_trend"],
            ratio_1h_avg             = rec["ratio_1h_avg"],
            ratio_increasing         = rec["ratio_increasing"],
            rule_triggered           = rec["rule_triggered"],
        ),
        chart_data  = chart_data,
        tx_costs    = [TxCost(**t) for t in savings["tx_savings"]],
        percentiles = {
            "p25": round(GLOBAL_P25, 8),
            "p75": round(GLOBAL_P75, 8),
            "p90": round(GLOBAL_P90, 8),
            "p99": round(GLOBAL_P99, 8),
            "floor": BASE_L2_FLOOR_GWEI,
        },
        chain_stats = {
            "floor_gwei":     BASE_L2_FLOOR_GWEI,
            "global_mean":    round(GLOBAL_MEAN, 8),
            "global_std":     round(GLOBAL_STD, 8),
            "z_score":        round(z_score, 3),
            "current_ratio":  round(gas_ratio or 0.5, 4),
            "roll_mean_6h":   round(roll_mean, 8),
            "roll_std_6h":    round(roll_std, 8),
        },
    )

@app.get("/api/metrics")
def get_metrics():
    return {
        "model_version":       MODEL_VERSION,
        "chain":               "Base L2",
        "trained_at":          REPORT.get("trained_at"),
        "architecture":        REPORT.get("architecture"),
        "features":            REPORT.get("features"),
        "mape_ok":             REPORT.get("mape_ok"),
        "metrics_overall":     REPORT.get("metrics_overall"),
        "ci_coverage_pct":     REPORT.get("ci_coverage_pct"),
        "percentiles":         REPORT.get("percentiles"),
    }

@app.get("/api/analysis")
def get_analysis():
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    current_fee, gas_ratio, ts_label, latest_features = get_current_data()

    if current_fee is None:
        current_fee = GLOBAL_MEAN
        
    roll_mean = latest_features.get("fee_rolling_mean", GLOBAL_MEAN)
    roll_std  = latest_features.get("fee_rolling_std", GLOBAL_STD)

    fc_fee   = forecast_fee(now_utc, latest_features)
    fc_ratio = forecast_ratio(now_utc, latest_features)
    z_score  = compute_fee_zscore(current_fee, roll_mean, roll_std)
    conf     = calculate_confidence(fc_fee)
    savings  = compute_savings_window(current_fee, fc_fee)
    
    rec      = run_rule_engine(
        current_fee=current_fee, fc_fee=fc_fee, fc_ratio=fc_ratio,
        current_ratio=gas_ratio or 0.5, z_score=z_score,
        confidence=conf, savings=savings, roll_std=roll_std
    )

    return {
        "model_version":  MODEL_VERSION,
        "analysis_ts_wib": (now_utc + WIB_OFFSET).strftime("%Y-%m-%d %H:%M:%S WIB"),

        "current_conditions": {
            "fee_gwei":         round(current_fee, 8),
            "fee_zone":         classify_zone(current_fee, z_score),
            "fee_z_score":      round(z_score, 3),
            "is_at_floor":      is_at_floor(current_fee),
            "gas_used_ratio":   round(gas_ratio or 0.5, 4),
            "roll_mean_6h":     round(roll_mean, 8),
            "roll_std_6h":      round(roll_std, 8),
        },

        "forecast_summary": {
            "horizon_hours":    24,
            "min_fee_gwei":     savings["min_fee_gwei"],
            "max_fee_gwei":     round(float(fc_fee["yhat_upper"].max()), 8),
            "avg_fee_gwei":     round(float(fc_fee["yhat"].mean()), 8),
            "optimal_wait_minutes": savings["optimal_wait_minutes"],
            "confidence_score": round(conf, 3),
            "ci_width_avg":     round(float((fc_fee["yhat_upper"] - fc_fee["yhat_lower"]).mean()), 8),
        },

        "savings_potential": {
            "savings_pct":              savings["savings_pct"],
            "savings_pct_conservative": savings["savings_pct_conservative"],
            "tx_breakdown":             savings["tx_savings"],
        },

        "rule_engine_state": {
            "action":         rec["action"],
            "rule_triggered": rec["rule_triggered"],
            "urgency_score":  rec["urgency_score"],
            "z_score":        rec["z_score"],
            "confidence":     rec["confidence_score"],
        },

        "historical_context": {
            "floor_gwei":           BASE_L2_FLOOR_GWEI,
            "global_p90_gwei":      round(GLOBAL_P90, 8),
            "global_p99_gwei":      round(GLOBAL_P99, 8),
            "model_ci_coverage":    REPORT.get("ci_coverage_pct", 0),
            "model_mape":           REPORT.get("metrics_overall", {}).get("mape", 0),
        },
    }

@app.get("/api/academic-evaluation")
def get_academic_evaluation():
    return {
        "model_version": MODEL_VERSION,
        "evaluation_timestamp": REPORT.get("trained_at"),
        "architecture": REPORT.get("architecture", "LightGBM"),
        "overall_metrics": REPORT.get("metrics_overall", {}),
        "spike_classification": {
            "precision": REPORT.get("metrics_overall", {}).get("precision", 0),
            "recall": REPORT.get("metrics_overall", {}).get("recall", 0),
            "f1_score": REPORT.get("metrics_overall", {}).get("f1", 0),
            "spike_mae": REPORT.get("metrics_overall", {}).get("spike_mae", 0),
        }
    }

@app.get("/api/status")
def get_status():
    data_ok        = os.path.exists(DATA_PATH)
    data_age_hours = None
    if data_ok:
        try:
            df = pd.read_csv(DATA_PATH, usecols=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
            data_age_hours  = round((datetime.utcnow() - df["timestamp"].max()).total_seconds() / 3600, 2)
        except Exception:
            pass

    mape = REPORT.get("metrics_overall", {}).get("mape", 999)
    return {
        "status":            "ok",
        "model_version":     MODEL_VERSION,
        "chain":             "Base L2",
        "fee_model_loaded":  True,
        "ratio_model_loaded": HAS_RATIO_MODEL,
        "model_degraded":    mape > MAPE_DEGRADED_THRESHOLD,
        "mape":              round(mape, 2),
        "ci_coverage_pct":   round(REPORT.get("ci_coverage_pct", 0), 1),
        "data_file_found":   data_ok,
        "data_age_hours":    data_age_hours,
        "data_stale":        data_age_hours is not None and data_age_hours > 1,
        "floor_gwei":        BASE_L2_FLOOR_GWEI,
        "p90_gwei":          round(GLOBAL_P90, 8),
        "p99_gwei":          round(GLOBAL_P99, 8),
        "server_time_wib":   (datetime.utcnow() + WIB_OFFSET).strftime("%Y-%m-%d %H:%M:%S"),
    }