"""
02_merge_weather_and_firms.py
==============================
Merges NASA POWER daily weather time series with FIRMS & Sipongi hotspot records,
mapped onto Uber H3 Resolution 8 peatland cells.

Computes:
  - Daily aggregated weather: T2M_MAX, RH2M, PRECTOTCORR, GWETTOP
  - Derived features:
      * days_without_rain (HTH: consecutive days with precipitation < 1.0 mm)
      * vapor_pressure_deficit (VPD in kPa from T2M_MAX & RH2M)
      * ground_water_level_cm (estimated GWL in cm, range 0 to -120 cm)
  - Nearest-neighbor weather mapping via cKDTree to H3 centroids
  - Target label: ignition_7d (1 if hotspot detected in cell or 1-ring within 7 days, else 0)

Output:
  data/processed/h3_training_dataset.parquet
"""

import os
import sys
import glob
import time
import logging
import gc
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree
import h3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MergeWeatherFirms")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
STATIC_PARQUET = os.path.join(PROCESSED_DIR, "h3_static_features.parquet")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "h3_training_dataset.parquet")

def calculate_vpd(temp_c, rh_percent):
    """
    Computes Vapor Pressure Deficit (VPD in kPa) using the Tetens formula.
    """
    temp_c = np.asarray(temp_c, dtype=np.float64)
    rh_percent = np.asarray(rh_percent, dtype=np.float64)
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_percent / 100.0)
    vpd = np.maximum(0.0, es - ea)
    return np.round(vpd, 3)

def calculate_gwl(gwettop, days_without_rain):
    """
    Estimates Peat Ground Water Level (GWL in cm).
    In tropical peat, GWETTOP=1.0 & dwr=0 -> ~ -10 cm (saturated).
    GWETTOP ~ 0.3 - 0.5 & dwr > 15 -> < -40 cm (critical PP 57/2016 threshold) down to -100 cm.
    Range: [-120.0, 0.0]
    """
    gwettop = np.asarray(gwettop, dtype=np.float64)
    dwr = np.asarray(days_without_rain, dtype=np.float64)
    soil_depletion = (1.0 - np.clip(gwettop, 0.0, 1.0)) * 72.0
    drought_subsidence = np.minimum(dwr, 35.0) * 1.5
    gwl = - (soil_depletion + drought_subsidence + 10.0)
    return np.round(np.clip(gwl, -120.0, 0.0), 2)

def load_nasa_weather():
    """
    Reads and merges all 20 NASA POWER daily regional CSV files.
    """
    nasa_dir = os.path.join(RAW_DIR, "nasa_power")
    csv_files = glob.glob(os.path.join(nasa_dir, "*.csv"))
    logger.info("Found %d NASA POWER CSV files in %s", len(csv_files), nasa_dir)

    dfs = []
    for f in csv_files:
        # Find line with -END HEADER-
        with open(f, 'r') as fp:
            skip = 0
            for i, line in enumerate(fp):
                if "-END HEADER-" in line:
                    skip = i + 1
                    break

        df = pd.read_csv(f, skiprows=skip)
        param_col = df.columns[-1]
        sub = df[["LAT", "LON", "YEAR", "DOY", param_col]].copy()
        sub["param"] = param_col.upper()
        sub.rename(columns={param_col: "val"}, inplace=True)
        # Filter invalid values (-999)
        sub.loc[sub["val"] <= -900, "val"] = np.nan
        dfs.append(sub)

    logger.info("Concatenating and pivoting parameters across %d files...", len(dfs))
    long_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Aggregate overlapping grid bounds
    grouped_long = long_df.groupby(["LAT", "LON", "YEAR", "DOY", "param"], as_index=False)["val"].mean()
    del long_df
    gc.collect()

    grouped = grouped_long.pivot(index=["LAT", "LON", "YEAR", "DOY"], columns="param", values="val").reset_index()
    del grouped_long
    gc.collect()

    # Convert YEAR + DOY to Date
    date_strs = grouped["YEAR"].astype(str) + "-" + grouped["DOY"].astype(str)
    grouped["date"] = pd.to_datetime(date_strs, format="%Y-%j").dt.date

    # Impute remaining missing values with regional climatological normals
    grouped["T2M_MAX"] = grouped["T2M_MAX"].fillna(31.0)
    grouped["RH2M"] = grouped["RH2M"].fillna(80.0)
    grouped["PRECTOTCORR"] = grouped["PRECTOTCORR"].fillna(0.0)
    grouped["GWETTOP"] = grouped["GWETTOP"].fillna(0.85)

    # Compute days without rain per (LAT, LON) time series
    logger.info("Computing HTH (days_without_rain) per weather grid point...")
    grouped.sort_values(by=["LAT", "LON", "date"], inplace=True)

    # Vectorized accumulation of days without rain per (LAT, LON) group
    is_dry = (grouped["PRECTOTCORR"] < 1.0).astype(int)
    # Reset count on rain
    grouped["days_without_rain"] = 0
    dwr_arr = np.zeros(len(grouped), dtype=np.int32)
    lats = grouped["LAT"].values
    lons = grouped["LON"].values
    dry_vals = is_dry.values

    current_dwr = 0
    last_loc = None
    for i in range(len(grouped)):
        loc = (lats[i], lons[i])
        if loc != last_loc:
            current_dwr = 0
            last_loc = loc
        if dry_vals[i] == 1:
            current_dwr += 1
        else:
            current_dwr = 0
        dwr_arr[i] = current_dwr

    grouped["days_without_rain"] = dwr_arr

    # Compute VPD and GWL
    grouped["vapor_pressure_deficit"] = calculate_vpd(grouped["T2M_MAX"], grouped["RH2M"])
    grouped["ground_water_level_cm"] = calculate_gwl(grouped["GWETTOP"], grouped["days_without_rain"])

    logger.info("Loaded %d daily weather records across %d unique grid points",
                len(grouped), grouped[["LAT", "LON"]].drop_duplicates().shape[0])
    return grouped

def load_all_hotspots(min_date, max_date, valid_h3_set):
    """
    Reads FIRMS shapefiles and Sipongi Excel files, filters to relevant dates,
    and returns a set of (h3_cell, date) representing fire events.
    """
    logger.info("Loading FIRMS & Sipongi hotspots between %s and %s...", min_date, max_date)
    hotspot_records = []

    # 1. FIRMS shapefiles
    firms_files = glob.glob(os.path.join(RAW_DIR, "firms", "**", "*.shp"), recursive=True)
    for f in firms_files:
        try:
            df = pyogrio.read_dataframe(
                f, columns=["LATITUDE", "LONGITUDE", "ACQ_DATE"],
                read_geometry=False, use_arrow=True
            )
            df = df[df["LONGITUDE"].between(95.0, 119.0) & df["LATITUDE"].between(-6.0, 5.0)]
            d = pd.to_datetime(df["ACQ_DATE"]).dt.date
            df = df[(d >= min_date) & (d <= max_date)].copy()
            df["date"] = pd.to_datetime(df["ACQ_DATE"]).dt.date
            hotspot_records.append(df[["LATITUDE", "LONGITUDE", "date"]])
        except Exception as e:
            logger.warning("Error reading FIRMS file %s: %s", os.path.basename(f), e)

    # 2. Sipongi Excel files
    sipongi_files = glob.glob(os.path.join(RAW_DIR, "sipongi", "sebaran-hotspot*.xlsx"))
    for f in sipongi_files:
        try:
            df = pd.read_excel(f, sheet_name="Detail")
            d = pd.to_datetime(df["Tanggal"], format="%d-%m-%Y", errors="coerce").dt.date
            mask = (d >= min_date) & (d <= max_date)
            sub = pd.DataFrame({
                "LATITUDE": pd.to_numeric(df.loc[mask, "Latitude"], errors="coerce"),
                "LONGITUDE": pd.to_numeric(df.loc[mask, "Longitude"], errors="coerce"),
                "date": d[mask]
            }).dropna()
            hotspot_records.append(sub)
        except Exception as e:
            logger.warning("Error reading Sipongi file %s: %s", os.path.basename(f), e)

    if not hotspot_records:
        logger.warning("No hotspots found!")
        return set()

    df_hotspots = pd.concat(hotspot_records, ignore_index=True).drop_duplicates()
    logger.info("Total raw hotspot records in date window: %d", len(df_hotspots))

    # Convert to H3 res 8 and expand to 1-ring neighbors (grid_disk 1)
    fire_events = set()
    for lat, lon, dt in zip(df_hotspots["LATITUDE"], df_hotspots["LONGITUDE"], df_hotspots["date"]):
        try:
            center_cell = h3.latlng_to_cell(lat, lon, 8)
            # Expand to 1-ring disk (cell + its 6 immediate neighbors)
            disk = h3.grid_disk(center_cell, 1)
            for neighbor in disk:
                if neighbor in valid_h3_set:
                    fire_events.add((neighbor, dt))
        except Exception:
            continue

    logger.info("Mapped into %d unique (h3_cell, date) peatland fire event associations", len(fire_events))
    return fire_events

def build_training_dataset():
    t0 = time.time()
    logger.info("=================================================================")
    logger.info("Nusantara Fire Sentinel: Step 02 - Merge Weather & FIRMS")
    logger.info("=================================================================")

    if not os.path.exists(STATIC_PARQUET):
        logger.error("Static H3 parquet not found: %s. Run 01_build_h3_static_grid.py first!", STATIC_PARQUET)
        sys.exit(1)

    df_static = pd.read_parquet(STATIC_PARQUET)
    logger.info("Loaded static H3 grid with %d cells", len(df_static))
    valid_h3_set = set(df_static["h3_index"])

    # Load weather data
    df_weather = load_nasa_weather()
    unique_dates = sorted(df_weather["date"].unique())
    min_date = unique_dates[0]
    max_date = unique_dates[-1]
    logger.info("Weather observation dates range from %s to %s (%d unique days)", min_date, max_date, len(unique_dates))

    # Load hotspots in date range
    fire_events = load_all_hotspots(min_date, max_date, valid_h3_set)

    # Pre-index fire events by cell: cell -> list of fire dates
    logger.info("Indexing fire dates per cell...")
    cell_fire_dates = {}
    for cell, f_date in fire_events:
        if cell not in cell_fire_dates:
            cell_fire_dates[cell] = []
        cell_fire_dates[cell].append(f_date)
    for cell in cell_fire_dates:
        cell_fire_dates[cell] = np.array(cell_fire_dates[cell], dtype="datetime64[D]")

    # Build cKDTree for weather mapping
    unique_weather_locs = df_weather[["LAT", "LON"]].drop_duplicates().values
    weather_tree = cKDTree(unique_weather_locs)
    
    h3_lats = df_static["centroid_lat"].values
    h3_lons = df_static["centroid_lon"].values
    h3_coords = np.column_stack([h3_lats, h3_lons])
    
    logger.info("Mapping %d H3 cells to nearest weather grid point...", len(df_static))
    _, nearest_idx = weather_tree.query(h3_coords, k=1, workers=-1)
    df_static["weather_lat"] = unique_weather_locs[nearest_idx, 0]
    df_static["weather_lon"] = unique_weather_locs[nearest_idx, 1]

    # Pre-index weather by (LAT, LON, date) for O(1) lookup
    logger.info("Indexing daily weather records...")
    df_weather.set_index(["LAT", "LON", "date"], inplace=True)
    weather_dict = df_weather[["ground_water_level_cm", "days_without_rain", "vapor_pressure_deficit"]].to_dict("index")

    # Static features dict
    static_dict = df_static.set_index("h3_index")[
        ["peat_depth_meters", "distance_to_road_meters", "distance_to_canal_meters", "region", "weather_lat", "weather_lon"]
    ].to_dict("index")

    # Generate training samples
    # We sample dates across the fire seasons: 2009, 2015, and 2026.
    # To maintain temporal balance and high resolution without runaway memory:
    # We include all dates for cells that experienced fire in that window,
    # plus periodic representative samples (every 3 days) for all cells.
    logger.info("Assembling dataset samples...")
    sample_dates = set(unique_dates[::3]) # baseline every 3 days
    # Also add the dates around any fire event
    for cell, f_dates in cell_fire_dates.items():
        for fd in f_dates:
            # Add dates within [fd - 7 days, fd]
            fd_ts = pd.Timestamp(fd)
            for delta in range(0, 8):
                prior_d = (fd_ts - pd.Timedelta(days=delta)).date()
                if prior_d in df_weather.index.levels[2]:
                    sample_dates.add(prior_d)

    sample_dates = sorted(list(sample_dates))
    logger.info("Selected %d sample dates for dataset assembly", len(sample_dates))

    records = []
    positive_count = 0
    negative_count = 0

    all_cells = df_static["h3_index"].values

    for d in sample_dates:
        d_np = np.datetime64(d, "D")
        d_plus_7 = d_np + np.timedelta64(7, "D")

        for cell in all_cells:
            # Check ignition_7d: any fire date in [d, d + 7]
            has_fire = 0
            if cell in cell_fire_dates:
                f_dates = cell_fire_dates[cell]
                # Check if any f_date falls in [d_np, d_plus_7]
                in_window = (f_dates >= d_np) & (f_dates <= d_plus_7)
                if np.any(in_window):
                    has_fire = 1

            # Subsampling of negatives to avoid 99.8% zero redundancy while preserving all positives
            if has_fire == 0:
                # Keep ~8% of negatives to achieve balanced ~1:8 or 1:10 positive/negative ratio
                if (hash((cell, d)) % 100) > 8:
                    continue
                negative_count += 1
            else:
                positive_count += 1

            s_info = static_dict[cell]
            w_key = (s_info["weather_lat"], s_info["weather_lon"], d)
            w_info = weather_dict.get(w_key)
            if w_info is None:
                continue

            records.append({
                "h3_index": cell,
                "date": d,
                "year": d.year,
                "month": d.month,
                "region": s_info["region"],
                "peat_depth_meters": s_info["peat_depth_meters"],
                "ground_water_level_cm": w_info["ground_water_level_cm"],
                "days_without_rain": int(w_info["days_without_rain"]),
                "vapor_pressure_deficit": w_info["vapor_pressure_deficit"],
                "distance_to_road_meters": s_info["distance_to_road_meters"],
                "distance_to_canal_meters": s_info["distance_to_canal_meters"],
                "ignition_7d": has_fire
            })

    df_train = pd.DataFrame(records)
    logger.info("Created dataset with %d records (Positives: %d, Negatives: %d, Positive Ratio: %.2f%%)",
                len(df_train), positive_count, negative_count, (positive_count / len(df_train)) * 100)

    # Save to Parquet
    df_train.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    logger.info("Successfully saved training dataset to %s (%.2f MB)", OUTPUT_PARQUET, file_size_mb)
    logger.info("Class distribution by year:\n%s", pd.crosstab(df_train["year"], df_train["ignition_7d"]).to_string())
    logger.info("Step 02 completed in %.2fs total", time.time() - t0)

if __name__ == "__main__":
    build_training_dataset()
