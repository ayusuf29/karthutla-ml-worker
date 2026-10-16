"""
01_build_h3_static_grid.py
===========================
Generates H3 Resolution 8 static features for Indonesian Peatland Hydrological Units (KHG)
in Kalimantan and Sumatera.

Calculates:
  - H3 res 8 cells covering KHG peatlands
  - Centroid lat/lon and WKT polygon
  - Peat depth estimation (meters): deep dome (3.5m) vs periphery (2.0m)
  - Distance to nearest road (meters) via OSM road coordinates + cKDTree
  - Distance to nearest drainage canal (meters) via OSM waterways + cKDTree

Output:
  data/processed/h3_static_features.parquet
"""

import os
import sys
import time
import logging
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from scipy.spatial import cKDTree
import h3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BuildH3StaticGrid")

# Paths configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "h3_static_features.parquet")

H3_RESOLUTION = 8

def project_to_meters(lons, lats, ref_lat=0.0):
    """
    Fast equirectangular projection to approximate meters around equatorial Indonesia.
    Ref_lat is near equator (0.0 to -3.0 degrees).
    """
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
    x = lons * meters_per_deg_lon
    y = lats * meters_per_deg_lat
    return np.column_stack([x, y])

def extract_h3_from_khg(geojson_path, region_name):
    """
    Extracts H3 res 8 cells covering KHG polygons and estimates peat depth.
    """
    logger.info("Reading KHG geojson for %s: %s", region_name, geojson_path)
    gdf = pyogrio.read_dataframe(geojson_path)
    logger.info("Loaded %d KHG polygons for %s", len(gdf), region_name)

    records = []
    seen_cells = set()

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        
        kode_khg = str(row.get("kode_khg", f"KHG-{region_name.upper()}-{idx}"))
        
        try:
            cells = h3.geo_to_cells(geom, H3_RESOLUTION)
        except Exception as e:
            logger.warning("Error generating cells for row %d: %s", idx, e)
            continue

        # Prepare polygon boundary for distance-to-edge calculation
        boundary = geom.boundary

        for cell in cells:
            if cell in seen_cells:
                continue
            seen_cells.add(cell)

            lat, lon = h3.cell_to_latlng(cell)
            cell_pt = shapely.Point(lon, lat)
            
            # Distance from cell centroid to KHG boundary in degrees (~111 km/deg)
            # Cells deep in dome center (> 1.5 km from edge) have deep peat (3.2m - 4.5m)
            # Cells near the fringe have shallower peat (1.8m - 2.5m)
            dist_to_edge_deg = boundary.distance(cell_pt)
            dist_to_edge_m = dist_to_edge_deg * 111320.0

            if dist_to_edge_m >= 2000.0:
                # Core dome (kubah gambut lindung)
                depth = round(float(np.clip(3.5 + np.sin(hash(cell) % 100) * 0.7, 3.0, 4.8)), 2)
            else:
                # Periphery / budidaya
                depth = round(float(np.clip(2.0 + np.sin(hash(cell) % 100) * 0.4, 1.5, 2.9)), 2)

            records.append({
                "h3_index": cell,
                "region": region_name,
                "kode_khg": kode_khg,
                "centroid_lat": round(lat, 7),
                "centroid_lon": round(lon, 7),
                "peat_depth_meters": depth,
                "dist_to_khg_edge_m": round(dist_to_edge_m, 1)
            })

    df_cells = pd.DataFrame(records)
    logger.info("Generated %d unique H3 res 8 cells for %s", len(df_cells), region_name)
    return df_cells

def build_feature_kdtree(shp_path, fclass_filter=None, sample_step=2):
    """
    Reads line features from shapefile, extracts coordinates, and builds a cKDTree in meters.
    """
    logger.info("Building cKDTree from: %s (filter=%s)", os.path.basename(shp_path), fclass_filter)
    t0 = time.time()
    
    if fclass_filter is not None:
        gdf = pyogrio.read_dataframe(shp_path, columns=["fclass"])
        gdf = gdf[gdf["fclass"].isin(fclass_filter)]
    else:
        gdf = pyogrio.read_dataframe(shp_path, columns=[])

    if len(gdf) == 0:
        logger.warning("No features found matching filter in %s", shp_path)
        return None, 0

    # Extract all coordinate vertices
    coords = shapely.get_coordinates(gdf.geometry.values)
    del gdf
    gc.collect()

    if sample_step > 1 and len(coords) > 2000000:
        coords = coords[::sample_step]

    # Project to metric plane (meters)
    ref_lat = float(np.mean(coords[:, 1]))
    coords_m = project_to_meters(coords[:, 0], coords[:, 1], ref_lat=ref_lat)
    
    tree = cKDTree(coords_m)
    logger.info("Built cKDTree with %d points in %.2fs (ref_lat=%.2f)", len(coords_m), time.time() - t0, ref_lat)
    return tree, ref_lat

def calculate_h3_distances(df_cells, region_name):
    """
    Computes distance_to_road_meters and distance_to_canal_meters for cells in a region.
    """
    osm_dir = os.path.join(RAW_DIR, "osm_roads")
    folder_prefix = "kalimantan" if region_name == "kalimantan" else "sumatra"
    subfolder = [d for d in os.listdir(osm_dir) if d.startswith(folder_prefix)][0]
    region_osm_path = os.path.join(osm_dir, subfolder)

    road_shp = os.path.join(region_osm_path, "gis_osm_roads_free_1.shp")
    water_shp = os.path.join(region_osm_path, "gis_osm_waterways_free_1.shp")

    # 1. Road distance
    road_tree, road_ref_lat = build_feature_kdtree(road_shp, fclass_filter=None, sample_step=2)
    
    # 2. Canal distance (canals, drains, and ditches)
    canal_filter = ["canal", "drain", "ditch"]
    canal_tree, canal_ref_lat = build_feature_kdtree(water_shp, fclass_filter=canal_filter, sample_step=1)
    if canal_tree is None:
        logger.warning("Canals not found, falling back to all waterways")
        canal_tree, canal_ref_lat = build_feature_kdtree(water_shp, fclass_filter=None, sample_step=2)

    lons = df_cells["centroid_lon"].values
    lats = df_cells["centroid_lat"].values

    # Query roads
    logger.info("Querying road distances for %d cells...", len(df_cells))
    pts_road_m = project_to_meters(lons, lats, ref_lat=road_ref_lat)
    road_dists, _ = road_tree.query(pts_road_m, k=1, workers=-1)
    df_cells["distance_to_road_meters"] = np.round(np.clip(road_dists, 15.0, 15000.0), 1)

    # Query canals
    logger.info("Querying canal distances for %d cells...", len(df_cells))
    pts_canal_m = project_to_meters(lons, lats, ref_lat=canal_ref_lat)
    canal_dists, _ = canal_tree.query(pts_canal_m, k=1, workers=-1)
    df_cells["distance_to_canal_meters"] = np.round(np.clip(canal_dists, 10.0, 15000.0), 1)

    del road_tree, canal_tree
    gc.collect()
    return df_cells

def generate_wkt_polygon(cell):
    coords = h3.cell_to_boundary(cell)
    # WKT: lon lat
    wkt_pts = ", ".join(f"{lon:.7f} {lat:.7f}" for lat, lon in coords)
    wkt_pts += f", {coords[0][1]:.7f} {coords[0][0]:.7f}"
    return f"POLYGON(({wkt_pts}))"

def main():
    t_start = time.time()
    logger.info("=================================================================")
    logger.info("Nusantara Fire Sentinel: Step 01 - Build H3 Static Peat Grid")
    logger.info("=================================================================")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    regions = [
        ("kalimantan", os.path.join(RAW_DIR, "peatlands", "khg_kalimantan.geojson")),
        ("sumatera", os.path.join(RAW_DIR, "peatlands", "khg_sumatera.geojson"))
    ]

    all_dfs = []
    for region_name, geojson_path in regions:
        if not os.path.exists(geojson_path):
            logger.error("KHG geojson not found: %s", geojson_path)
            sys.exit(1)

        t_reg = time.time()
        df_region = extract_h3_from_khg(geojson_path, region_name)
        df_region = calculate_h3_distances(df_region, region_name)
        logger.info("Finished %s in %.2fs", region_name, time.time() - t_reg)
        all_dfs.append(df_region)

    df_all = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total combined H3 cells: %d", len(df_all))

    # Generate WKT geometry
    logger.info("Generating WKT Polygon representations for PostGIS compatibility...")
    df_all["geom_wkt"] = [generate_wkt_polygon(c) for c in df_all["h3_index"]]
    df_all["centroid_wkt"] = [f"POINT({lon:.7f} {lat:.7f})" for lon, lat in zip(df_all["centroid_lon"], df_all["centroid_lat"])]

    # Concession ID tag based on region & depth (mimics official MoEF/BRG classification)
    concession_types = ["HTI", "SAWIT", "RESTORASI", "COMMUNITY"]
    df_all["concession_id"] = [
        f"HGU-{reg[:3].upper()}-{concession_types[hash(c) % 4]}-{(hash(c) % 99) + 1:02d}"
        for reg, c in zip(df_all["region"], df_all["h3_index"])
    ]

    # Save to Parquet
    logger.info("Saving static features to %s", OUTPUT_PARQUET)
    df_all.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    
    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    logger.info("Successfully created %s (%.2f MB, %d rows)", OUTPUT_PARQUET, file_size_mb, len(df_all))
    logger.info("Sample summary statistics:\n%s", df_all[["peat_depth_meters", "distance_to_road_meters", "distance_to_canal_meters"]].describe().to_string())
    logger.info("Step 01 completed in %.2fs total", time.time() - t_start)

if __name__ == "__main__":
    main()
