import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

BASE_RPC_URL     = "https://mainnet.base.org"
SAVE_PATH        = "model/gas_fee_historical.csv"
DAYS_HISTORY     = 30
INTERVAL_MINUTES = 5
MAX_WORKERS      = 5
BATCH_SAVE_SIZE  = 100

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_latest_block_number():
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        resp = requests.post(BASE_RPC_URL, json=payload, timeout=10)
        data = resp.json()
        if 'result' not in data:
            log(f"[ERROR] Invalid RPC Response: {data}")
            return None
        return int(data['result'], 16)
    except Exception as e:
        log(f"[ERROR] RPC failure: {e}")
        return None

def get_block_by_number(block_number):
    payload = {"jsonrpc": "2.0", "method": "eth_getBlockByNumber", "params": [hex(block_number), False], "id": 1}
    try:
        resp = requests.post(BASE_RPC_URL, json=payload, timeout=10)
        block = resp.json().get('result')
        if not block:
            return None

        base_fee_wei = int(block.get('baseFeePerGas', '0x0'), 16)
        gas_used = int(block.get('gasUsed', '0x0'), 16)
        gas_limit = int(block.get('gasLimit', '0x1'), 16)
        timestamp_unix = int(block.get('timestamp', '0x0'), 16)

        return {
            'block_number': block_number,
            'timestamp': datetime.fromtimestamp(timestamp_unix, tz=timezone.utc),
            'base_fee_wei': base_fee_wei,
            'base_fee_gwei': round(base_fee_wei / 1e9, 8),
            'gas_used': gas_used,
            'gas_limit': gas_limit,
            'gas_used_ratio': round(gas_used / gas_limit, 4) if gas_limit > 0 else 0,
        }
    except Exception:
        return None

def main():
    log("Initializing Base L2 Historical Data Ingestion...")
    log("Calibrating Base L2 block time...")

    latest_block = get_latest_block_number()
    if not latest_block:
        log("Gagal mendapatkan block terbaru. Keluar.")
        return

    sample_blocks = [get_block_by_number(latest_block - i) for i in [0, 100, 500]]
    sample_blocks = [b for b in sample_blocks if b is not None]

    if len(sample_blocks) >= 2:
        time_diff = (sample_blocks[0]['timestamp'] - sample_blocks[-1]['timestamp']).total_seconds()
        block_diff = sample_blocks[0]['block_number'] - sample_blocks[-1]['block_number']
        avg_block_time = time_diff / block_diff if block_diff > 0 else 2.0
    else:
        avg_block_time = 2.0

    log(f"Average block time calibrated at {avg_block_time:.3f} seconds.")

    blocks_per_interval = int((INTERVAL_MINUTES * 60) / avg_block_time)
    total_intervals = int(DAYS_HISTORY * 24 * 60 / INTERVAL_MINUTES)

    start_interval = 0
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    if os.path.exists(SAVE_PATH):
        try:
            existing_df = pd.read_csv(SAVE_PATH)
            if not existing_df.empty:
                min_block = existing_df['block_number'].min()
                start_interval = int((latest_block - min_block) / blocks_per_interval) + 1
                log(f"Found existing dataset. Resuming extraction from block {min_block}.")
        except Exception as e:
            log(f"[WARNING] Failed to read existing CSV. Starting fresh. Error: {e}")

    intervals_to_fetch = total_intervals - start_interval

    if intervals_to_fetch <= 0:
        log("Target history duration already fulfilled. No new data to fetch.")
        return

    log(f"Target intervals to fetch: {intervals_to_fetch:,}")
    log("Initiating concurrent data extraction...")

    target_blocks = [latest_block - ((start_interval + i) * blocks_per_interval) for i in range(intervals_to_fetch)]
    target_blocks = [b for b in target_blocks if b > 0]

    fetched_records = []
    failed_requests = 0
    processed_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_block = {executor.submit(get_block_by_number, block): block for block in target_blocks}

        for future in as_completed(future_to_block):
            result = future.result()
            if result:
                fetched_records.append(result)
            else:
                failed_requests += 1

            processed_count += 1

            if len(fetched_records) >= BATCH_SAVE_SIZE:
                df_batch = pd.DataFrame(fetched_records)
                write_header = not os.path.exists(SAVE_PATH)
                df_batch.to_csv(SAVE_PATH, mode='a', index=False, header=write_header)
                fetched_records.clear()

                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta_seconds = (len(target_blocks) - processed_count) / rate
                log(f"Progress: {processed_count:,}/{len(target_blocks):,} blocks | Rate: {rate:.1f} req/s | ETA: {eta_seconds/60:.1f} min")

    if fetched_records:
        df_batch = pd.DataFrame(fetched_records)
        write_header = not os.path.exists(SAVE_PATH)
        df_batch.to_csv(SAVE_PATH, mode='a', index=False, header=write_header)

    total_time = time.time() - start_time
    log(f"Extraction complete. Time elapsed: {total_time/60:.2f} minutes.")
    log(f"Failed requests: {failed_requests}")

    log("Post-processing and cleaning dataset...")
    final_df = pd.read_csv(SAVE_PATH)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    final_df = final_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    final_df['hour'] = final_df['timestamp'].dt.hour
    final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
    final_df['date'] = final_df['timestamp'].dt.date

    final_df = final_df[(final_df['base_fee_gwei'] > 0) & (final_df['base_fee_gwei'] < 1000)]
    final_df.to_csv(SAVE_PATH, index=False)
    log("Process successfully terminated. Dataset is ready for Prophet ML model.")

if __name__ == "__main__":
    main()
