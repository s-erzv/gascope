import time
import pandas as pd
import os
from ingest_base_data import get_latest_block_number, get_block_by_number

DATA_PATH = "model/gas_fee_historical.csv"
POLL_INTERVAL = 10

def start_pulse():
    last_block = None
    print(f"📡 Streamer Active: Pumping real-time data to {DATA_PATH}...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    while True:
        try:
            current_block = get_latest_block_number()
            if current_block and current_block != last_block:
                data = get_block_by_number(current_block)
                if data:

                    ts = data['timestamp']
                    data['hour'] = ts.hour
                    data['day_of_week'] = ts.weekday()
                    data['date'] = ts.date()

                    df = pd.DataFrame([data])
                    write_header = not os.path.exists(DATA_PATH)
                    df.to_csv(DATA_PATH, mode='a', index=False, header=write_header)
                    last_block = current_block
                    print(f"✔️ Block {current_block} Synced | Fee: {data['base_fee_gwei']:.5f} Gwei")
        except Exception as e:
            print(f"❌ Streamer Error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    start_pulse()
