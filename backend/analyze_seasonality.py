import pandas as pd
from prophet.serialize import model_from_json

def extract_seasonality():

    with open("model/prophet_fee_model.json", "r") as f:
        m = model_from_json(f.read())

    future = pd.DataFrame({
        'ds': pd.date_range(start='2026-05-09 00:00:00', periods=288, freq='5min')
    })

    future['gas_used_ratio'] = 0.5

    forecast = m.predict(future)

    daily = forecast[['ds', 'daily']].copy()
    daily['ds_wib'] = daily['ds'] + pd.Timedelta(hours=7)
    daily['time_wib'] = daily['ds_wib'].dt.strftime('%H:%M WIB')

    top_spike = daily.sort_values('daily', ascending=False).head(8)
    top_cheap = daily.sort_values('daily', ascending=True).head(8)

    print("\n" + "="*50)
    print(" 🔴 WAKTU RAWAN SPIKE (Gas Paling Mahal di Base L2)")
    print("="*50)
    for _, row in top_spike.iterrows():
        print(f" Jam {row['time_wib']} -> Impact: +{row['daily']:.6f} Gwei")

    print("\n" + "="*50)
    print(" 🟢 WAKTU PALING MURAH (Waktu Ideal Eksekusi)")
    print("="*50)
    for _, row in top_cheap.iterrows():
        print(f" Jam {row['time_wib']} -> Impact: {row['daily']:.6f} Gwei")
    print("\n")

if __name__ == "__main__":
    extract_seasonality()
