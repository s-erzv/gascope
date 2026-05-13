import sqlite3
import time
import logging
import os
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE_RPC_URL     = "https://mainnet.base.org"
DB_PATH          = "model/gas_fee.db"
DAYS_HISTORY     = 90
INTERVAL_SECONDS = 300
MAX_WORKERS      = 4
BATCH_COMMIT     = 200

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ingest")

def get_latest_block_number() -> int | None:
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        r = requests.post(BASE_RPC_URL, json=payload, timeout=10)
        return int(r.json()["result"], 16)
    except Exception as e:
        log.error(f"eth_blockNumber error: {e}")
        return None

def get_block_by_number(block_number: int, max_retries: int = 3) -> dict | None:
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), False],
        "id": 1
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(BASE_RPC_URL, json=payload, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            block = r.json().get("result")
            if not block:
                return None

            base_fee_wei = int(block.get("baseFeePerGas", "0x0"), 16)
            gas_used     = int(block.get("gasUsed", "0x0"), 16)
            gas_limit    = int(block.get("gasLimit", "0x1"), 16)
            ts_unix      = int(block.get("timestamp", "0x0"), 16)

            return {
                "block_number":   block_number,
                "timestamp":      datetime.fromtimestamp(ts_unix, tz=timezone.utc),
                "base_fee_wei":   base_fee_wei,
                "base_fee_gwei":  round(base_fee_wei / 1e9, 8),
                "gas_used":       gas_used,
                "gas_limit":      gas_limit,
                "gas_used_ratio": round(gas_used / gas_limit, 4) if gas_limit > 0 else 0.0,
            }
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            return None
    return None

def init_db(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON gas_fee(interval_ts)")
    con.commit()
    con.close()

def get_oldest_ts(con: sqlite3.Connection) -> datetime | None:
    row = con.execute("SELECT MIN(interval_ts) FROM gas_fee").fetchone()
    if row and row[0]:
        return datetime.fromisoformat(row[0].replace("Z", "+00:00"))
    return None

def insert_batch(con: sqlite3.Connection, records: list[tuple]):
    con.executemany("""
        INSERT OR REPLACE INTO gas_fee 
        (interval_ts, block_number, base_fee_gwei, base_fee_wei, gas_used, gas_limit, gas_used_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    con.commit()

def main():
    init_db(DB_PATH)
    log.info("Mulai ingest historis Base L2...")

    latest_block = get_latest_block_number()
    assert latest_block, "Gagal ambil latest block"

    samples = [get_block_by_number(latest_block - i) for i in [0, 200, 1000]]
    samples = [s for s in samples if s]
    if len(samples) >= 2:
        dt  = (samples[0]["timestamp"] - samples[-1]["timestamp"]).total_seconds()
        db  = samples[0]["block_number"] - samples[-1]["block_number"]
        avg_block_time = dt / db if db > 0 else 2.0
    else:
        avg_block_time = 2.0

    blocks_per_interval = max(1, int(INTERVAL_SECONDS / avg_block_time))
    log.info(f"Block time: {avg_block_time:.2f}s | Blocks/interval: {blocks_per_interval}")

    con = sqlite3.connect(DB_PATH)
    oldest_ts = get_oldest_ts(con)
    con.close()

    now_utc = datetime.now(timezone.utc)
    latest_block_data = get_block_by_number(latest_block)
    latest_ts = latest_block_data["timestamp"] if latest_block_data else now_utc

    target_start = now_utc - timedelta(days=DAYS_HISTORY)

    if oldest_ts and oldest_ts <= target_start + timedelta(hours=1):
        log.info(f"Data sudah cukup (oldest: {oldest_ts.isoformat()}). Skip.")
        return

    fetch_from = target_start
    fetch_to   = oldest_ts if oldest_ts else now_utc

    intervals = []
    t = fetch_from
    while t < fetch_to:
        ts_epoch = int(t.timestamp())
        rounded  = ts_epoch - (ts_epoch % INTERVAL_SECONDS)
        intervals.append(datetime.fromtimestamp(rounded, tz=timezone.utc))
        t += timedelta(seconds=INTERVAL_SECONDS)

    if not intervals:
        log.info("Tidak ada interval baru yang perlu di-fetch.")
        return

    log.info(f"Fetch {len(intervals):,} interval dari {intervals[0]} s/d {intervals[-1]}")

    def interval_to_target_block(interval_ts: datetime) -> int:
        delta_sec   = (latest_ts - interval_ts).total_seconds()
        blocks_back = int(delta_sec / avg_block_time)
        return max(1, latest_block - blocks_back)

    target_blocks = [(iv, interval_to_target_block(iv)) for iv in intervals]

    records_buffer = []
    failed = 0
    done   = 0
    start  = time.time()

    con = sqlite3.connect(DB_PATH)

    def fetch_task(iv_block):
        iv, block = iv_block
        data = get_block_by_number(block)
        if data and 0 < data["base_fee_gwei"] < 1000:
            return (
                iv.strftime("%Y-%m-%dT%H:%M:%SZ"),
                data["block_number"],
                round(data["base_fee_gwei"], 8),
                data["base_fee_wei"],
                data["gas_used"],
                data["gas_limit"],
                round(data["gas_used_ratio"], 4),
            )
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_task, tb): tb for tb in target_blocks}
        for future in as_completed(futures):
            result = future.result()
            done  += 1
            if result:
                records_buffer.append(result)
            else:
                failed += 1

            if len(records_buffer) >= BATCH_COMMIT:
                insert_batch(con, records_buffer)
                records_buffer.clear()
                elapsed = time.time() - start
                rate    = done / elapsed
                eta     = (len(target_blocks) - done) / rate if rate > 0 else 0
                log.info(f"Progress: {done:,}/{len(target_blocks):,} | {rate:.1f} req/s | ETA {eta/60:.1f} min")

    if records_buffer:
        insert_batch(con, records_buffer)

    con.close()
    log.info(f"Selesai. Failed: {failed} | Waktu: {(time.time()-start)/60:.1f} menit")

if __name__ == "__main__":
    main()
