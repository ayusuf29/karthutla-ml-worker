#!/usr/bin/env bash
set -e

# Navigate to ml-worker directory
cd "$(dirname "$0")"

echo "==============================================================================="
echo "  NUSANTARA FIRE SENTINEL - END-TO-END ML TRAINING PIPELINE (CPU LAPTOP)"
echo "==============================================================================="
echo ""

# Detect python executable
if command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[ERROR] Python is not installed or not in PATH!"
    exit 1
fi

echo "[1/4] Running Step 01: Build H3 Resolution 8 Static Grid (KHG + OSM)..."
$PY_BIN -u training/01_build_h3_static_grid.py

echo ""
echo "[2/5] Running Step 02: Merge NASA POWER Weather and FIRMS Telemetry..."
$PY_BIN -u training/02_merge_weather_and_firms.py

echo ""
echo "[3/5] Running Step 05: Real-Time Telemetry Scraping (May 2026 - Present)..."
$PY_BIN -u training/05_realtime_data_fetcher.py --start 2026-05-01 || echo "[WARNING] Step 05 realtime fetcher warning, proceeding with available dataset."

echo ""
echo "[4/5] Running Step 03: Train Karhutla XGBoost Classifier..."
$PY_BIN -u training/03_train_xgboost.py

echo ""
echo "[5/5] Running Step 04: Seed Real 2026 Predictions to PostGIS..."
$PY_BIN -u training/04_seed_real_2026_to_postgis.py

echo ""
echo "==============================================================================="
echo "  PIPELINE SUCCESSFUL: All 4 ML pipeline stages completed!"
echo "  Model: ml-worker/models/karhutla_xgb_v1.json"
echo "  Metrics: ml-worker/models/feature_metrics.json"
echo "  PostGIS: peat_h3_grid and h3_daily_risk_snapshot seeded with real 2026 data."
echo "==============================================================================="
