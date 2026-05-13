#!/bin/bash
# entrypoint.sh — Startup sequence untuk HF Spaces
#
# Urutan:
#   1. Ingest data historis (kalau DB kosong / kurang dari 7 hari)
#   2. Train model (kalau model hilang atau lebih dari 7 hari lama)
#   3. Jalankan streamer di background
#   4. Jalankan API server di foreground

set -e

DB_PATH="model/gas_fee.db"
REPORT_PATH="model/training_report.json"
MODEL_PATH="model/lgbm_fee_median.txt"
PROFILES_PATH="model/hourly_profiles.json"
SPIKE_CLF_PATH="model/lgbm_spike_clf.txt"

echo "══════════════════════════════════════"
echo "  Gascope v4 — HF Spaces Startup"
echo "══════════════════════════════════════"

# ── Step 1: Cek apakah perlu ingest ───────────────────────────
ROW_COUNT=0
if [ -f "$DB_PATH" ]; then
    ROW_COUNT=$(python3 -c "
import sqlite3
try:
    con = sqlite3.connect('$DB_PATH')
    r = con.execute('SELECT COUNT(*) FROM gas_fee').fetchone()
    print(r[0])
except:
    print(0)
")
fi

echo "Rows in DB: $ROW_COUNT"

# Ingest kalau kurang dari 7 hari data (7×24×12 = 2016 rows)
if [ "$ROW_COUNT" -lt 2016 ]; then
    echo "▶ Ingest historis (90 hari)..."
    python3 ingest_base_data.py
    echo "✔ Ingest selesai"
else
    echo "✔ Data historis cukup, skip ingest"
fi

# ── Step 2: Cek apakah perlu training ─────────────────────────
NEED_TRAIN=false

# Train jika model utama atau profiles tidak ada
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$REPORT_PATH" ] || [ ! -f "$PROFILES_PATH" ]; then
    echo "Model atau profiles tidak ditemukan, perlu training"
    NEED_TRAIN=true
fi

# Re-train kalau model lebih dari 7 hari lama
if [ -f "$REPORT_PATH" ] && [ "$NEED_TRAIN" = false ]; then
    MODEL_AGE_DAYS=$(python3 -c "
import json, datetime
try:
    r = json.load(open('$REPORT_PATH'))
    trained_at = datetime.datetime.fromisoformat(r['trained_at'])
    age = (datetime.datetime.utcnow() - trained_at).days
    print(age)
except:
    print(999)
")
    echo "Model age: ${MODEL_AGE_DAYS} days"
    if [ "$MODEL_AGE_DAYS" -gt 7 ]; then
        NEED_TRAIN=true
    fi
fi

if [ "$NEED_TRAIN" = true ]; then
    echo "▶ Training model..."
    python3 train_model.py
    echo "✔ Training selesai"
else
    echo "✔ Model masih fresh, skip training"
fi

# ── Step 3: Jalankan streamer di background ───────────────────
echo "▶ Menjalankan streamer di background..."
python3 streamer.py &
STREAMER_PID=$!
echo "✔ Streamer PID: $STREAMER_PID"

# ── Step 4: Jalankan API server ───────────────────────────────
echo "▶ Menjalankan API server di port 7860..."
echo "══════════════════════════════════════"
exec uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
