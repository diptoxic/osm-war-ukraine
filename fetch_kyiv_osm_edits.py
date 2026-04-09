#!/usr/bin/env python3
"""
Fetch OSM building contributions for Kyiv with exact timestamps.
Outputs a GeoJSON ready for QGIS spatial analysis.

Usage:
    pip install requests geopandas shapely pandas
    python fetch_kyiv_osm_edits.py
"""

import requests
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Kyiv bounding box (lon_min, lat_min, lon_max, lat_max)
KYIV_BBOX = "30.2,50.2,30.9,50.7"

# Period of interest: start of war → 3 months
START = "2022-02-01"
END   = "2022-12-31"

# Ohsome endpoint for individual contributions with centroids
URL_CENTROID = "https://api.ohsome.org/v1/contributions/centroid"

OUTPUT_DIR = "outputs_kyiv_1y"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_kyiv_contributions():
    """
    Fetches OSM building contributions (centroid + timestamp) for Kyiv.
    Returns GeoDataFrame with columns: geometry, timestamp, edit_type, osm_id
    """
    print(f"Fetching OSM contributions for Kyiv ({START} → {END})...")

    params = {
        "bboxes":  KYIV_BBOX,
        "time":    f"{START},{END}",
        "filter":  "building=* and type:way",
        "timeout": "300",
    }

    r = requests.post(URL_CENTROID, data=params, timeout=320)
    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text[:300]}")
        return None

    data = r.json()
    features = data.get("features", [])
    print(f"  → {len(features)} contributions found")

    if not features:
        return None

    records = []
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry") or {}
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            records.append({
                "geometry":       Point(lon, lat),
                "timestamp":      props.get("@timestamp"),
                "osm_id":         props.get("@osmId"),
                "contribution_type": props.get("@contributionTypes", "unknown"),
            })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    gdf["date"]      = gdf["timestamp"].dt.date.astype(str)

    return gdf


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gdf = fetch_kyiv_contributions()

    if gdf is None or gdf.empty:
        print("No data retrieved. Check your bounding box and dates.")
        exit(1)

    # Save as GeoJSON for QGIS
    out_path = os.path.join(OUTPUT_DIR, "kyiv_osm_edits_2022.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved: {out_path}")
    print(f"Total edits: {len(gdf)}")
    print(f"Date range:  {gdf['timestamp'].min()} → {gdf['timestamp'].max()}")

    # Also save as CSV for inspection
    csv_path = os.path.join(OUTPUT_DIR, "kyiv_osm_edits_2022.csv")
    gdf.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"Saved CSV:   {csv_path}")