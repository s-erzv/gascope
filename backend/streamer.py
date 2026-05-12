import time
import pandas as pd
import os
from datetime import datetime, timezone
from ingest_base_data import get_latest_block_number, get_block_by_number

DATA_PATH = "model/gas_fee_historical.csv"

def start_pulse():
    print(f"📡 Streamer Active: Pumping real-time data to {DATA_PATH}")
    print("⏳ Menyelaraskan ritme data (Interval: 5 Menit / 300 detik)...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Pastikan urutan kolom sesuai 100% dengan ingest_base_data.py
    TARGET_COLUMNS = [
        'block_number', 'timestamp', 'base_fee_wei', 'base_fee_gwei', 
        'gas_used', 'gas_limit', 'gas_used_ratio', 'hour', 'day_of_week', 'date'
    ]

    while True:
        try:
            current_block = get_latest_block_number()
            if current_block:
                data = get_block_by_number(current_block)
                if data:
                    ts = data['timestamp']
                    data['hour'] = ts.hour
                    data['day_of_week'] = ts.weekday()
                    data['date'] = ts.date()

                    # Konversi ke DataFrame dengan urutan kolom ketat (Strict Schema)
                    df = pd.DataFrame([data])[TARGET_COLUMNS]
                    
                    write_header = not os.path.exists(DATA_PATH)
                    df.to_csv(DATA_PATH, mode='a', index=False, header=write_header)
                    
                    log_time = datetime.now().strftime('%H:%M:%S')
                    print(f"[{log_time}] ✔️ Block {current_block} Synced | Fee: {data['base_fee_gwei']:.5f} Gwei")
        
        except Exception as e:
            print(f"Streamer Error: {e}")

        # Smart Sleep: Hitung detik menuju kelipatan 5 menit selanjutnya (Cron alignment)
        # Ini memastikan data kita rapi di menit :00, :05, :10 tanpa drift.
        now = datetime.now()
        seconds_to_next_5min = 300 - ((now.minute % 5) * 60 + now.second)
        
        # Tambahkan sedikit buffer 2 detik agar tidak terlalu awal saat bangun
        sleep_duration = seconds_to_next_5min + 2 
        
        time.sleep(max(1, sleep_duration))

if __name__ == "__main__":
    start_pulse()