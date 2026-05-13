import time
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from ingest_base_data import get_latest_block_number, get_block_by_number

DB_PATH          = "model/gas_fee.db"
INTERVAL_SECONDS = 300
AVG_BLOCK_TIME   = 2.0
BLOCKS_PER_STEP  = int(INTERVAL_SECONDS / AVG_BLOCK_TIME)
MAX_RETRY        = 3

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("streamer")

def init_db(path: str):
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS gas_fee (
            interval_ts TEXT PRIMARY KEY,
            block_number INTEGER,
            base_fee_gwei REAL,
            base_fee_wei INTEGER,
            gas_used INTEGER,
            gas_limit INTEGER,
            gas_used_ratio REAL
        )
    """)
    con.commit()
    con.close()

def round_to_interval(dt: datetime, interval_sec: int = INTERVAL_SECONDS) -> datetime:
    ts = int(dt.timestamp())
    return datetime.fromtimestamp(ts - (ts % interval_sec), tz=timezone.utc)

def target_block_for_interval(interval_ts: datetime, latest_block: int, latest_ts: datetime) -> int:
    delta_sec = (latest_ts - interval_ts).total_seconds()
    blocks_back = int(delta_sec / AVG_BLOCK_TIME)
    return max(1, latest_block - blocks_back)

def fetch_with_retry(block_number: int) -> dict | None:
    for attempt in range(MAX_RETRY):
        data = get_block_by_number(block_number)
        if data:
            return data
        wait = 2 ** attempt
        log.warning(f"Retry {attempt+1}/{MAX_RETRY} block {block_number}, tunggu {wait}s")
        time.sleep(wait)
    return None

def insert_record(con: sqlite3.Connection, interval_ts: datetime, data: dict):
    ratio = data["gas_used_ratio"]
    fee   = data["base_fee_gwei"]

    if not (0 < fee < 1000):
        raise ValueError(f"Fee tidak valid: {fee}")
    if not (0 <= ratio <= 1):
        raise ValueError(f"Ratio tidak valid: {ratio}")

    con.execute("""
        INSERT OR REPLACE INTO gas_fee 
        (interval_ts, block_number, base_fee_gwei, base_fee_wei, gas_used, gas_limit, gas_used_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        interval_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        data["block_number"],
        round(fee, 8),
        data["base_fee_wei"],
        data["gas_used"],
        data["gas_limit"],
        round(ratio, 4),
    ))
    con.commit()

def seconds_until_next_interval(interval_sec: int = INTERVAL_SECONDS) -> float:
    now_ts = datetime.now(timezone.utc).timestamp()
    next_ts = (int(now_ts / interval_sec) + 1) * interval_sec
    return max(1.0, (next_ts - now_ts) + 3.0)

def main():
    import os
    os.makedirs("model", exist_ok=True)
    init_db(DB_PATH)
    log.info(f"Streamer aktif  {DB_PATH} | interval {INTERVAL_SECONDS}s")

    con = sqlite3.connect(DB_PATH)

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            interval_ts = round_to_interval(now_utc)

            latest_block = get_latest_block_number()
            if latest_block is None:
                raise RuntimeError("Gagal ambil latest block")

            latest_data = fetch_with_retry(latest_block)
            if latest_data is None:
                raise RuntimeError(f"Gagal ambil data block {latest_block}")

            latest_block_ts = latest_data["timestamp"]

            target_block = target_block_for_interval(interval_ts, latest_block, latest_block_ts)
            data = fetch_with_retry(target_block)
            if data is None:
                raise RuntimeError(f"Gagal ambil target block {target_block}")

            insert_record(con, interval_ts, data)

            log.info(
                f" {interval_ts.strftime('%H:%M')} UTC | "
                f"block {target_block} | "
                f"fee {data['base_fee_gwei']:.5f} Gwei | "
                f"ratio {data['gas_used_ratio']:.3f}"
            )

        except Exception as e:
            log.error(f"Streamer error: {e}")

        sleep_sec = seconds_until_next_interval()
        time.sleep(sleep_sec)

if __name__ == "__main__":
    main()
