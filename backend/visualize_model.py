import pandas as pd
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("Memuat model Prophet...")
try:
    with open('model/prophet_gas_model.json', 'r') as fin:
        model = model_from_json(fin.read())
except Exception as e:
    print(f"Error memuat model: {e}")
    exit()

print("Mengkalkulasi proyeksi 24 jam ke depan...")

future = model.make_future_dataframe(periods=288, freq='5min')
forecast = model.predict(future)

print("Menggambar Grafik 1: Proyeksi Utama (Forecast Plot)...")
fig1 = model.plot(forecast, figsize=(12, 6))
plt.title('Proyeksi Machine Learning: Base L2 Gas Fee', fontsize=14, fontweight='bold')
plt.ylabel('Gas Fee (Gwei)', fontsize=12)
plt.xlabel('Waktu (Timestamp)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('grafik_proyeksi_utama.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

print("Menggambar Grafik 2: Dekomposisi Komponen (Component Plot)...")
fig2 = model.plot_components(forecast, figsize=(12, 8))
plt.savefig('grafik_komponen_model.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

print("✅ Selesai! Dua grafik telah berhasil disimpan: 'grafik_proyeksi_utama.png' dan 'grafik_komponen_model.png'.")
