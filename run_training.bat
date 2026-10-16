@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===============================================================================
echo   NUSANTARA FIRE SENTINEL - END-TO-END ML TRAINING PIPELINE (CPU LAPTOP)
echo ===============================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    exit /b 1
)

echo [1/4] Running Step 01: Build H3 Resolution 8 Static Grid (KHG + OSM)...
python -u training\01_build_h3_static_grid.py
if %errorlevel% neq 0 (
    echo [FAILED] Step 01 encountered an error. Aborting pipeline.
    exit /b 1
)
echo.

echo [2/4] Running Step 02: Merge NASA POWER Weather and FIRMS Telemetry...
python -u training\02_merge_weather_and_firms.py
if %errorlevel% neq 0 (
    echo [FAILED] Step 02 encountered an error. Aborting pipeline.
    exit /b 1
)
echo.

echo [3/4] Running Step 03: Train Karhutla XGBoost Classifier...
python -u training\03_train_xgboost.py
if %errorlevel% neq 0 (
    echo [FAILED] Step 03 encountered an error. Aborting pipeline.
    exit /b 1
)
echo.

echo [4/4] Running Step 04: Seed Real 2026 Predictions to PostGIS...
python -u training\04_seed_real_2026_to_postgis.py
if %errorlevel% neq 0 (
    echo [FAILED] Step 04 encountered an error. Aborting pipeline.
    exit /b 1
)
echo.

echo ===============================================================================
echo   PIPELINE SUCCESSFUL: All 4 ML pipeline stages completed!
echo   Model: ml-worker\models\karhutla_xgb_v1.json
echo   Metrics: ml-worker\models\feature_metrics.json
echo   PostGIS: peat_h3_grid and h3_daily_risk_snapshot seeded with real 2026 data.
echo ===============================================================================
