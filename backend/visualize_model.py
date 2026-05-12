import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# 1. Konfigurasi Awal
DATA_PATH = "model/gas_fee_historical.csv"
BASE_L2_FLOOR_GWEI = 0.005

print("Memuat model LightGBM...")
try:
    model_lower  = lgb.Booster(model_file='model/lgbm_fee_lower.txt')
    model_median = lgb.Booster(model_file='model/lgbm_fee_median.txt')
    model_upper  = lgb.Booster(model_file='model/lgbm_fee_upper.txt')
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    exit()

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Data {DATA_PATH} tidak ditemukan.")
    exit()

print("Mengambil state jaringan terakhir...")
df_raw = pd.read_csv(DATA_PATH).tail(300)
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True).dt.tz_localize(None)

window = 72
shifted_fee = df_raw["base_fee_gwei"].shift(1)
df_raw["fee_lag_1"] = shifted_fee
df_raw["fee_rolling_mean"] = shifted_fee.rolling(window, min_periods=6).mean()
df_raw["fee_rolling_std"]  = shifted_fee.rolling(window, min_periods=6).std()
df_raw["fee_rolling_p75"]  = shifted_fee.rolling(window, min_periods=6).quantile(0.75)

if "gas_used_ratio" in df_raw.columns:
    df_raw["gas_used_ratio"] = df_raw["gas_used_ratio"].fillna(0.5)
    shifted_ratio = df_raw["gas_used_ratio"].shift(1)
    df_raw["ratio_lag_1"] = shifted_ratio
    df_raw["ratio_rolling_mean"] = shifted_ratio.rolling(window, min_periods=6).mean()
    df_raw["ratio_rolling_std"]  = shifted_ratio.rolling(window, min_periods=6).std()

latest = df_raw.iloc[-1]
last_timestamp = latest["timestamp"]

FEATURES = ["hour", "day_of_week", "fee_lag_1", "fee_rolling_mean", "fee_rolling_std", "fee_rolling_p75", "ratio_lag_1", "ratio_rolling_mean", "ratio_rolling_std"]

print("Mengkalkulasi proyeksi 24 jam ke depan...")
future_dates = pd.date_range(start=last_timestamp + timedelta(minutes=5), periods=288, freq='5min')
future = pd.DataFrame({"ds": future_dates})
future["hour"] = future["ds"].dt.hour
future["day_of_week"] = future["ds"].dt.dayofweek

for feat in FEATURES:
    if feat not in ["hour", "day_of_week"]:
        future[feat] = latest[feat]

X_pred = future[FEATURES]

future["yhat_lower"] = np.clip(model_lower.predict(X_pred), BASE_L2_FLOOR_GWEI, None)
future["yhat_median"] = np.clip(model_median.predict(X_pred), BASE_L2_FLOOR_GWEI, None)
future["yhat_upper"] = np.clip(model_upper.predict(X_pred), BASE_L2_FLOOR_GWEI, None)

print("Menggambar Grafik 1: Proyeksi Utama (Forecast Plot)...")
plt.figure(figsize=(14, 7))
plt.plot(future["ds"], future["yhat_median"], label='Median Prediction (P50)', color='#1f77b4', linewidth=2)
plt.fill_between(future["ds"], future["yhat_lower"], future["yhat_upper"], color='#1f77b4', alpha=0.25, label='Confidence Interval (P10 - P90)')
plt.title('LightGBM Projection: Base L2 Gas Fee (Next 24h)', fontsize=14, fontweight='bold')
plt.ylabel('Gas Fee (Gwei)', fontsize=12)
plt.xlabel('Timestamp (UTC)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('grafik_proyeksi_utama.png', dpi=300)
plt.close()

print("Menggambar Grafik 2: Feature Importances...")
importances = model_median.feature_importance(importance_type='split')
feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importances}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='#2ca02c', edgecolor='black', alpha=0.8)
plt.title('LightGBM Decision Matrix: Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score (Split Counts)', fontsize=12)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('grafik_komponen_model.png', dpi=300)
plt.close()

print("✅ Selesai! Dua grafik telah berhasil disimpan: 'grafik_proyeksi_utama.png' dan 'grafik_komponen_model.png'.")