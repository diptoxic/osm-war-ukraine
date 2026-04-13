#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OSM UNDER THE TEST OF WAR — Generalized data collection script           ║
║   OSM Contributions × ACLED Bombings                                       ║
║   Full war duration: Feb. 2022 → present                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Project [anonymized] / [anonymized]                                  ║
║  Sponsor: Raphaël Bres                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  CONFIGURABLE ZONES (STUDY_AREAS):                                         ║
║    kyiv      → Kyiv city + close suburbs                                   ║
║    kharkiv   → Kharkiv (occupied zone / eastern front line)                ║
║    donetsk   → Donbas / Donetsk Oblast                                     ║
║    kherson   → Kherson (occupation then liberation Nov. 2022)              ║
║    mariupol  → Mariupol (siege Mar–May 2022)                               ║
║    ukraine   → Entire Ukraine (national aggregation)                       ║
║                                                                             ║
║  OUTPUTS (per zone, in outputs/<zone>/):                                   ║
║    contributions_<zone>.geojson   ← 1 point / real OSM contribution       ║
║    deletions_<zone>.geojson       ← deletions only                         ║
║    ruins_<zone>.geojson           ← buildings tagged ruins/destroyed       ║
║    acled_bombings_<zone>.geojson  ← filtered ACLED bombings                ║
║    match_osm_acled_<zone>.geojson ← enriched contributions (ACLED corr.)  ║
║    series_mensuelle_<zone>.csv    ← monthly aggregation for charts         ║
║                                                                             ║
║  ACLED BOMBING FILTER (corrected):                                         ║
║    event_type    = "Explosions/Remote violence"                            ║
║    sub_event_type ∈ air/drone strike, shelling/artillery/missile attack,   ║
║                     remote explosive/landmine/IED, missile attack,         ║
║                     guided missile strike                                  ║
║                                                                             ║
║  DEPENDENCIES:                                                              ║
║    pip install requests geopandas shapely pandas numpy matplotlib          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Point, shape, box as sbox
from shapely.ops import unary_union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# ── War period ────────────────────────────────────────────────────────────────
WAR_START = "2022-02-24"
WAR_END   = "2024-12-31"   # adjust if needed

# ── Study zones ───────────────────────────────────────────────────────────────
# Each zone: (bbox "lon_min,lat_min,lon_max,lat_max", readable name)
STUDY_AREAS = {
    "kyiv":     ("30.20,50.20,30.90,50.60", "Kyiv (city)"),
    "kharkiv":  ("36.00,49.80,36.60,50.20", "Kharkiv"),
    "donetsk":  ("37.50,47.80,38.20,48.20", "Donetsk"),
    "kherson":  ("32.40,46.50,33.00,46.80", "Kherson"),
    "mariupol": ("37.40,47.00,37.80,47.20", "Mariupol"),
    "ukraine":  ("22.00,44.00,40.50,52.50", "Entire Ukraine"),
}

# Zone(s) to process — modify this list to target a specific zone
# E.g.: ZONES_TO_PROCESS = ["kyiv", "kharkiv"]
ZONES_TO_PROCESS = ["donetsk"]

# ── ACLED file ────────────────────────────────────────────────────────────────
ACLED_FILE = "dataACLED.shp"   # place at project root

# ── Directories ───────────────────────────────────────────────────────────────
BASE_OUTPUT = "outputs"
BASE_CACHE  = "data/cache"

# ── Ohsome parameters ─────────────────────────────────────────────────────────
FILTER_BUILDINGS = "building=* and type:way"
FILTER_RUINS     = (
    "(building=ruins or ruins=yes or ruins=* "
    "or destroyed:building=* or disused:building=*) and type:way"
)
OHSOME_URL = "https://api.ohsome.org/v1"

# ── Spatio-temporal correlation parameters ────────────────────────────────────
SPATIAL_BUFFER_M = 1000   # radius in meters
WINDOW_DAYS      = 7      # temporal window post-strike
UTM_CRS          = "EPSG:32637"   # UTM 37N — centered on Ukraine

# ── ACLED bombing filter ──────────────────────────────────────────────────────
# event_type must contain "Explosion"
ACLED_EVENT_TYPE_FILTER = "Explosion"

# sub_event_type must match one of these types
ACLED_BOMBING_SUBTYPES = [
    "air/drone strike",
    "shelling/artillery/missile attack",
    "remote explosive/landmine/ied",
    "missile attack",
    "guided missile strike",
]

# ── Monthly slicing for ohsome (avoids timeouts) ─────────────────────────────
def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Generates (month_start, month_end) pairs between start and end."""
    ranges = []
    cur = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        next_m = (cur + pd.offsets.MonthEnd(1)).replace(hour=0,
                                                         minute=0,
                                                         second=0,
                                                         microsecond=0)
        # The last chunk closes exactly on WAR_END
        end_m = min(next_m, end_ts)
        ranges.append((cur.strftime("%Y-%m-%d"), end_m.strftime("%Y-%m-%d")))
        cur = next_m + pd.Timedelta(days=1)
    return ranges

MONTH_RANGES = _month_ranges(WAR_START, WAR_END)


# =============================================================================
# UTILITAIRES
# =============================================================================

def _strip_tz(s: pd.Series) -> pd.Series:
    try:
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def ohsome_post(endpoint: str, params: dict, retries: int = 3) -> dict | None:
    url = f"{OHSOME_URL}/{endpoint}"
    for i in range(retries):
        try:
            r = requests.post(url, data=params, timeout=200)
            if r.status_code == 200:
                return r.json()
            log.warning(f"  ohsome HTTP {r.status_code} ({i+1}/{retries}): "
                        f"{r.text[:150]}")
        except Exception as e:
            log.warning(f"  ohsome network error ({i+1}/{retries}): {e}")
        time.sleep(6 * (i + 1))
    return None


def geom_to_centroid(geom_dict: dict) -> Point | None:
    try:
        g = shape(geom_dict)
        c = g.centroid
        return c if not c.is_empty else None
    except Exception:
        return None


def cache_path(zone: str, name: str) -> str:
    d = os.path.join(BASE_CACHE, zone)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.json")


def output_path(zone: str, name: str) -> str:
    d = os.path.join(BASE_OUTPUT, zone)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_geojson(gdf: gpd.GeoDataFrame, path: str):
    """Clean save: serialize datetime columns before export."""
    tmp = gdf.copy()
    for col in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[col]):
            tmp[col] = tmp[col].astype(str)
        elif tmp[col].dtype == object:
            # Convert lists/dicts to str for GeoJSON
            try:
                tmp[col] = tmp[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict)) else x
                )
            except Exception:
                pass
    tmp.to_file(path, driver="GeoJSON")


# =============================================================================
# 1. OSM CONTRIBUTIONS — REAL GEOMETRIES (contributions/geometry)
# =============================================================================

def fetch_contributions(zone: str, bbox: str,
                        start: str = WAR_START,
                        end:   str = WAR_END) -> gpd.GeoDataFrame:
    """
    Retrieves real geometries of OSM contributions (buildings)
    month by month via contributions/geometry.

    Each returned feature = a modified/created/deleted building,
    with its centroid, date, and contribution type.

    Output columns:
      geometry      : WGS84 Point (building centroid)
      osm_id        : OSM identifier
      contrib_type  : creation | modification | deletion
      timestamp     : naive UTC datetime
      date          : date only (str YYYY-MM-DD)
      month         : monthly period (str YYYY-MM)
      building      : building tag value
      lon / lat     : decimal coordinates
    """
    out_file = output_path(zone, f"contributions_{zone}.geojson")
    if os.path.exists(out_file):
        log.info(f"[{zone}] contributions → cache: {out_file}")
        gdf = gpd.read_file(out_file)
        gdf["timestamp"] = pd.to_datetime(gdf.get("timestamp"), errors="coerce")
        return gdf

    log.info(f"[{zone}] Fetching contributions {start}→{end} "
             f"({len(MONTH_RANGES)} months)…")

    month_ranges = _month_ranges(start, end)
    all_features = []

    for start_m, end_m in month_ranges:
        c_key  = f"contrib_{start_m[:7]}"
        c_file = cache_path(zone, c_key)

        if os.path.exists(c_file):
            features = json.load(open(c_file))
            log.info(f"  [{zone}] {start_m[:7]} → cache ({len(features)} ft.)")
        else:
            log.info(f"  [{zone}] {start_m[:7]} — ohsome query…")
            params = {
                "bboxes":     bbox,
                "time":       f"{start_m},{end_m}",
                "filter":     FILTER_BUILDINGS,
                "timeout":    "180",
                "properties": "metadata,tags",
            }
            resp     = ohsome_post("contributions/geometry", params)
            features = resp.get("features", []) if resp else []
            json.dump(features, open(c_file, "w"))
            log.info(f"  [{zone}] {start_m[:7]} → {len(features)} features")
            time.sleep(1.0)

        all_features.extend(features)

    log.info(f"[{zone}] Total raw features: {len(all_features)}")
    if not all_features:
        log.error(f"[{zone}] No contributions — check bbox and period.")
        return gpd.GeoDataFrame()

    records = _parse_contribution_features(all_features)
    if not records:
        return gpd.GeoDataFrame()

    gdf = _to_geodataframe(records)
    save_geojson(gdf, out_file)
    log.info(f"[{zone}] ✔ {len(gdf)} contributions → {out_file}")
    _log_contrib_summary(gdf, zone)
    return gdf


def fetch_deletions(zone: str, bbox: str,
                    start: str = WAR_START,
                    end:   str = WAR_END) -> gpd.GeoDataFrame:
    """
    Retrieves OSM deletions via Ohsome (filtering on @deletion on the response side).
    """

    out_file = output_path(zone, f"deletions_{zone}.geojson")

    # Cache
    if os.path.exists(out_file):
        log.info(f"[{zone}] deletions → cache")
        gdf = gpd.read_file(out_file)
        gdf["timestamp"] = pd.to_datetime(gdf.get("timestamp"), errors="coerce")
        return gdf

    log.info(f"[{zone}] Fetching deletions {start}→{end}…")

    month_ranges = _month_ranges(start, end)
    all_features = []

    for start_m, end_m in month_ranges:
        c_file = cache_path(zone, f"del_{start_m[:7]}")

        # Monthly cache
        if os.path.exists(c_file):
            with open(c_file) as f:
                features = json.load(f)
        else:
            params = {
                "bboxes": bbox,
                "time": f"{start_m},{end_m}",
                "filter": FILTER_BUILDINGS,
                "properties": "tags,metadata",
                "timeout": "600",
            }

            resp = ohsome_post("contributions/geometry", params)
            features = resp.get("features", []) if resp else []

            with open(c_file, "w") as f:
                json.dump(features, f)

            time.sleep(1.0)

        all_features.extend(features)

    # Filter deletions here
    deleted_features = []
    for f in all_features:
        props = f.get("properties", {})

        if props.get("@deletion") is True:
            deleted_features.append(f)

    if not deleted_features:
        log.info(f"[{zone}] ⚠ no deletions found")
        return gpd.GeoDataFrame()

    # Transform to internal format
    records = _parse_contribution_features(
        deleted_features,
        force_type="deletion"
    )

    if not records:
        return gpd.GeoDataFrame()

    gdf = _to_geodataframe(records)

    # Handle timestamp
    if "timestamp" in gdf.columns:
        gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], errors="coerce")

    save_geojson(gdf, out_file)

    log.info(f"[{zone}] ✔ {len(gdf)} deletions → {out_file}")

    return gdf


def fetch_ruins(zone: str, bbox: str) -> gpd.GeoDataFrame:
    """
    Snapshots of buildings tagged ruins/destroyed at key dates.
    Covers the full war duration with quarterly snapshots.
    """
    out_file = output_path(zone, f"ruins_{zone}.geojson")
    if os.path.exists(out_file):
        log.info(f"[{zone}] ruins → cache")
        return gpd.read_file(out_file)

    # Quarterly snapshots across the full war period
    start_ts = pd.Timestamp(WAR_START)
    end_ts   = pd.Timestamp(WAR_END)
    key_dates = []
    cur = start_ts
    while cur <= end_ts:
        key_dates.append(cur.strftime("%Y-%m-%d"))
        cur += pd.DateOffset(months=3)

    log.info(f"[{zone}] Ruins — {len(key_dates)} quarterly snapshots…")
    all_features = []

    for d in key_dates:
        c_file = cache_path(zone, f"ruins_{d}")
        if os.path.exists(c_file):
            features = json.load(open(c_file))
        else:
            params = {
                "bboxes":     bbox,
                "time":       d,
                "filter":     FILTER_RUINS,
                "timeout":    "120",
                "properties": "metadata,tags",
            }
            resp     = ohsome_post("elements/geometry", params)
            features = resp.get("features", []) if resp else []
            for f in features:
                f["snapshot_date"] = d
            json.dump(features, open(c_file, "w"))
            log.info(f"  [{zone}] ruins {d} → {len(features)}")
            time.sleep(0.5)
        all_features.extend(features)

    if not all_features:
        return gpd.GeoDataFrame()

    records = []
    for feat in all_features:
        props = feat.get("properties", {})
        c     = geom_to_centroid(feat.get("geometry"))
        if not c:
            continue
        records.append({
            "geometry":      c,
            "snapshot_date": feat.get("snapshot_date", ""),
            "osm_id":        str(props.get("@osmId") or ""),
            "building":      str(props.get("building", "")),
            "ruins":         str(props.get("ruins", "")),
            "lon":           round(c.x, 6),
            "lat":           round(c.y, 6),
        })

    if not records:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(out_file, driver="GeoJSON")
    log.info(f"[{zone}] ✔ {len(gdf)} ruins → {out_file}")
    return gdf


# ── Helpers parsing ───────────────────────────────────────────────────────────

def _parse_contribution_features(features: list,
                                  force_type: str = None) -> list:
    records = []
    for feat in features:
        props = feat.get("properties", {})
        c     = geom_to_centroid(feat.get("geometry"))
        if not c:
            continue
        ts_raw = (props.get("@timestamp") or
                  props.get("timestamp")  or
                  props.get("@toTimestamp") or "")
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
        if pd.notna(ts):
            ts = ts.tz_localize(None)
        ct = force_type or (
            props.get("@contributionTypes") or
            props.get("contributionType")   or
            "modification"
        )
        records.append({
            "geometry":     c,
            "osm_id":       str(props.get("@osmId") or props.get("osmId") or ""),
            "contrib_type": str(ct),
            "timestamp":    ts,
            "date":         ts.strftime("%Y-%m-%d") if pd.notna(ts) else "",
            "month":        ts.strftime("%Y-%m")    if pd.notna(ts) else "",
            "building":     str(props.get("building", "yes")),
            "lon":          round(c.x, 6),
            "lat":          round(c.y, 6),
        })
    return records


def _to_geodataframe(records: list) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], errors="coerce")
    return gdf


def _log_contrib_summary(gdf: gpd.GeoDataFrame, zone: str):
    for ct in ["creation", "modification", "deletion"]:
        n = (gdf["contrib_type"] == ct).sum()
        log.info(f"  [{zone}] {ct:<14} : {n:>7,}")


# =============================================================================
# 2. ACLED — BOMBINGS ONLY (corrected filter)
# =============================================================================

def load_acled_bombings(zone: str, bbox: str,
                        start: str = WAR_START,
                        end:   str = WAR_END) -> gpd.GeoDataFrame:
    """
    Loads ACLED events and filters ONLY bombings:
      event_type    ∋  "Explosion"    (Explosions/Remote violence)
      sub_event_type ∈  ACLED_BOMBING_SUBTYPES

    Returns a GeoDataFrame with normalized columns:
      geometry, event_date, event_type, sub_event_type,
      fatalities, location, notes
    """
    out_file = output_path(zone, f"acled_bombings_{zone}.geojson")
    if os.path.exists(out_file):
        log.info(f"[{zone}] ACLED bombings → cache")
        gdf = gpd.read_file(out_file)
        gdf["event_date"] = pd.to_datetime(gdf.get("event_date"), errors="coerce")
        return gdf

    if not os.path.exists(ACLED_FILE):
        log.warning(f"ACLED file not found: {ACLED_FILE}")
        return gpd.GeoDataFrame()

    log.info(f"[{zone}] Loading ACLED: {ACLED_FILE}")
    gdf = gpd.read_file(ACLED_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # ── Bbox filter ───────────────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()

    # ── Date normalization ────────────────────────────────────────────────────
    date_col = next(
        (c for c in gdf.columns
         if "event_date" in c or ("date" in c and "event" in c)), None
    ) or next((c for c in gdf.columns if "date" in c), None)

    if not date_col:
        log.warning(f"[{zone}] Date column not found in ACLED")
        return gpd.GeoDataFrame()

    gdf["event_date"] = _strip_tz(
        pd.to_datetime(gdf[date_col], errors="coerce", utc=True)
    )

    # ── Period filter ─────────────────────────────────────────────────────────
    gdf = gdf[
        (gdf["event_date"] >= pd.Timestamp(start)) &
        (gdf["event_date"] <= pd.Timestamp(end))
    ].copy()

    # ── Filter event_type = "Explosions/Remote violence" ─────────────────────
    event_col = next(
        (c for c in gdf.columns if c == "event_type"), None
    )
    if event_col:
        before = len(gdf)
        gdf = gdf[
            gdf[event_col].str.contains(
                ACLED_EVENT_TYPE_FILTER, case=False, na=False
            )
        ].copy()
        log.info(f"[{zone}] Filter event_type 'Explosion': "
                 f"{before} → {len(gdf)} events")
    else:
        log.warning(f"[{zone}] event_type column not found — filter disabled")

    # ── Filter sub_event_type ─────────────────────────────────────────────────
    sub_col = next(
        (c for c in gdf.columns if "sub_event" in c), None
    )
    if sub_col:
        pattern = "|".join(ACLED_BOMBING_SUBTYPES)
        before  = len(gdf)
        gdf = gdf[
            gdf[sub_col].str.lower().str.contains(pattern, na=False)
        ].copy()
        log.info(f"[{zone}] Filter sub_event_type bombings: "
                 f"{before} → {len(gdf)} events")
    else:
        log.warning(f"[{zone}] sub_event_type column not found")

    if gdf.empty:
        log.warning(f"[{zone}] No ACLED bombings after filtering.")
        return gpd.GeoDataFrame()

    # ── Output column normalization ───────────────────────────────────────────
    fatalities_col = next(
        (c for c in gdf.columns if "fatal" in c or "death" in c), None
    )
    location_col = next(
        (c for c in gdf.columns if "location" in c or "admin" in c), None
    )
    notes_col = next(
        (c for c in gdf.columns if "notes" in c or "source" in c), None
    )

    gdf_out = gpd.GeoDataFrame({
        "geometry":       gdf.geometry,
        "event_date":     gdf["event_date"],
        "event_type":     gdf[event_col]    if event_col  else "",
        "sub_event_type": gdf[sub_col]      if sub_col    else "",
        "fatalities":     pd.to_numeric(
                              gdf[fatalities_col], errors="coerce"
                          ).fillna(0).astype(int) if fatalities_col else 0,
        "location":       gdf[location_col] if location_col else "",
        "notes":          gdf[notes_col]    if notes_col  else "",
    }, crs="EPSG:4326").reset_index(drop=True)

    tmp = gdf_out.copy()
    tmp["event_date"] = tmp["event_date"].astype(str)
    tmp.to_file(out_file, driver="GeoJSON")
    log.info(f"[{zone}] ✔ {len(gdf_out)} bombings → {out_file}")
    return gdf_out


# =============================================================================
# 3. SPATIO-TEMPORAL CROSS-CORRELATION OSM × BOMBINGS
# =============================================================================
# READING DIRECTION (specific to the war context):
#
#   Starting from OSM CONTRIBUTIONS (point by point), we check
#   whether a bombing occurred in the WINDOW_DAYS days PRECEDING
#   the edit, within SPATIAL_BUFFER_M meters.
#
#   Hypothesis: an OSM contributor who edits a building may do so
#   in response to a recent bombing in their immediate neighborhood.
#
#   Produced field: has_bombing_Xkm_Yd
#     1 = a bombing occurred within X km in the Y days before the edit
#     0 = no nearby/recent bombing
#
#   Complementary fields:
#     n_bombings_Xkm_Yd  : number of bombings within the window
#     dist_nearest_m     : distance to nearest bombing (meters)
#     nearest_sub_type   : sub-type of the nearest bombing
#     fatalities_nearby  : total casualties within the window

def match_contributions_with_bombings(
    gdf_osm:    gpd.GeoDataFrame,
    gdf_acled:  gpd.GeoDataFrame,
    zone:       str,
    buffer_m:   int = SPATIAL_BUFFER_M,
    window_days: int = WINDOW_DAYS,
) -> gpd.GeoDataFrame:

    out_file = output_path(zone, f"match_osm_acled_{zone}.geojson")

    if os.path.exists(out_file):
        log.info(f"[{zone}] match OSM×ACLED → cache")
        return gpd.read_file(out_file)

    if gdf_osm.empty or gdf_acled.empty:
        log.warning(f"[{zone}] Missing data for cross-correlation.")
        return gpd.GeoDataFrame()

    log.info(f"[{zone}] Cross-correlation: {len(gdf_osm):,} OSM × {len(gdf_acled):,} ACLED")

    # Metric projection
    osm_utm   = gdf_osm.to_crs(UTM_CRS).copy()
    acled_utm = gdf_acled.to_crs(UTM_CRS).copy()

    # Timestamps
    osm_dates = _strip_tz(pd.to_datetime(gdf_osm["timestamp"], errors="coerce"))
    acled_dates = _strip_tz(pd.to_datetime(gdf_acled["event_date"], errors="coerce"))

    sub_type_col = "sub_event_type" if "sub_event_type" in gdf_acled.columns else None
    fatal_col    = "fatalities" if "fatalities" in gdf_acled.columns else None

    field_name   = f"has_bombing_{buffer_m//1000}km_{window_days}d"
    n_field_name = f"n_bombings_{buffer_m//1000}km_{window_days}d"

    results = []

    for i, (idx, row_osm) in enumerate(osm_utm.iterrows()):

        if i % 5000 == 0 and i > 0:
            log.info(f"  {i:,}/{len(osm_utm):,} processed…")

        edit_date = osm_dates.iloc[i]

        # Default values
        nearest_sub = None
        fatal_nearby = 0
        min_dist = None
        n_in_buf = 0

        if pd.isna(edit_date):
            results.append({
                field_name: 0,
                n_field_name: 0,
                "dist_nearest_m": None,
                "nearest_sub_type": None,
                "fatalities_nearby": 0,
            })
            continue

        # Temporal window (BEFORE contribution)
        window_start = edit_date - timedelta(days=window_days)

        mask_time = (acled_dates >= window_start) & (acled_dates <= edit_date)
        acled_window = acled_utm[mask_time]

        if acled_window.empty:
            results.append({
                field_name: 0,
                n_field_name: 0,
                "dist_nearest_m": None,
                "nearest_sub_type": None,
                "fatalities_nearby": 0,
            })
            continue

        # Distances to ACLED events
        dists = acled_window.geometry.distance(row_osm.geometry)

        if not dists.empty:
            min_dist = float(dists.min())

            # Bombings within buffer
            in_buf = dists[dists <= buffer_m]
            n_in_buf = len(in_buf)

            # Nearest event
            if sub_type_col:
                nearest_idx = dists.idxmin()
                nearest_sub = str(acled_window.loc[nearest_idx, sub_type_col])

            # Casualties
            if fatal_col and n_in_buf > 0:
                idx_in = in_buf.index
                fatal_nearby = int(
                    pd.to_numeric(
                        gdf_acled.loc[gdf_acled.index.isin(idx_in), fatal_col],
                        errors="coerce"
                    ).fillna(0).sum()
                )

        results.append({
            field_name: int(n_in_buf > 0),
            n_field_name: n_in_buf,
            "dist_nearest_m": round(min_dist, 1) if min_dist else None,
            "nearest_sub_type": nearest_sub,
            "fatalities_nearby": fatal_nearby,
        })

    # Merge results
    res_df = pd.DataFrame(results, index=gdf_osm.index)
    gdf_out = gdf_osm.copy()

    for col in res_df.columns:
        gdf_out[col] = res_df[col].values

    save_geojson(gdf_out, out_file)

    log.info(f"[{zone}] ✔ cross-correlation complete → {out_file}")

    return gdf_out

# =============================================================================
# 4. MONTHLY TIME SERIES
# =============================================================================

def build_monthly_series(zone: str,
                          gdf_contrib: gpd.GeoDataFrame,
                          gdf_acled:   gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregates by month:
      n_contrib_total       : all contributions
      n_deletions           : deletions only
      n_modifications       : modifications only
      n_creations           : creations only
      n_bombings            : ACLED bombings
      fatalities            : total casualties
    Saved as CSV for use in QGIS / charts.
    """
    out_file = output_path(zone, f"series_mensuelle_{zone}.csv")
    if os.path.exists(out_file):
        log.info(f"[{zone}] Monthly series → cache")
        return pd.read_csv(out_file, parse_dates=["month"])

    rows = []
    start_ts = pd.Timestamp(WAR_START)
    end_ts   = pd.Timestamp(WAR_END)
    cur      = start_ts.replace(day=1)

    contrib_ts = _strip_tz(
        pd.to_datetime(gdf_contrib.get("timestamp"), errors="coerce")
    ) if not gdf_contrib.empty else pd.Series([], dtype="datetime64[ns]")

    contrib_types = gdf_contrib.get(
        "contrib_type", pd.Series([""] * len(gdf_contrib))
    ) if not gdf_contrib.empty else pd.Series([], dtype=str)

    acled_ts   = _strip_tz(
        pd.to_datetime(gdf_acled.get("event_date"), errors="coerce")
    ) if not gdf_acled.empty else pd.Series([], dtype="datetime64[ns]")

    fatal_col = "fatalities" if (
        not gdf_acled.empty and "fatalities" in gdf_acled.columns
    ) else None

    while cur <= end_ts:
        next_m = cur + pd.offsets.MonthEnd(1)
        mask_c = (contrib_ts >= cur) & (contrib_ts <= next_m) \
                 if not contrib_ts.empty else pd.Series([], dtype=bool)
        mask_a = (acled_ts >= cur) & (acled_ts <= next_m) \
                 if not acled_ts.empty else pd.Series([], dtype=bool)

        sub_c = gdf_contrib[mask_c] if (
            not gdf_contrib.empty and not mask_c.empty
        ) else gpd.GeoDataFrame()
        sub_a = gdf_acled[mask_a] if (
            not gdf_acled.empty and not mask_a.empty
        ) else gpd.GeoDataFrame()

        ct = sub_c.get("contrib_type", pd.Series()) if not sub_c.empty else pd.Series()

        rows.append({
            "month":            cur.strftime("%Y-%m"),
            "n_contrib_total":  len(sub_c),
            "n_deletions":      int((ct == "deletion").sum()),
            "n_modifications":  int((ct == "modification").sum()),
            "n_creations":      int((ct == "creation").sum()),
            "n_bombings":       len(sub_a),
            "fatalities":       int(
                pd.to_numeric(sub_a[fatal_col], errors="coerce").fillna(0).sum()
            ) if (fatal_col and not sub_a.empty) else 0,
        })
        cur = (next_m + pd.Timedelta(days=1)).replace(day=1)

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    log.info(f"[{zone}] ✔ Monthly series ({len(df)} months) → {out_file}")
    return df


# =============================================================================
# 5. SUMMARY CHARTS
# =============================================================================

def plot_series(zone: str, zone_name: str, df: pd.DataFrame):
    """
    Summary figure for a zone:
    - Total OSM activity per month (blue bars)
    - Deletions per month (red bars)
    - ACLED bombings (purple line)
    """
    if df.empty:
        return

    os.makedirs(output_path(zone, ""), exist_ok=True)
    df = df.copy()
    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()

    ax1.bar(df["month_dt"], df["n_contrib_total"],
            width=25, color="#3498DB", alpha=0.55, label="OSM contributions (total)")
    ax1.bar(df["month_dt"], df["n_deletions"],
            width=25, color="#E74C3C", alpha=0.85, label="OSM deletions")

    if df["n_bombings"].sum() > 0:
        ax2.plot(df["month_dt"], df["n_bombings"],
                 color="#8E44AD", lw=2.2, marker="o", ms=5,
                 label="ACLED bombings")
        ax2.fill_between(df["month_dt"], df["n_bombings"],
                         alpha=0.10, color="#8E44AD")
        ax2.set_ylabel("ACLED bombings / month",
                       color="#8E44AD", fontsize=10)

    # Milestones
    ax1.axvline(pd.Timestamp("2022-02-24"), color="red", lw=1.8,
                ls="--", alpha=0.8, label="Invasion Feb. 24, 2022")

    ax1.set_ylabel("OSM Contributions / Deletions / month",
                   color="#2C3E50", fontsize=10)
    ax1.set_title(
        f"OSM Activity × ACLED Bombings — {zone_name}\n"
        f"War duration: {WAR_START} → {WAR_END}",
        fontsize=13, fontweight="bold"
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")
    ax1.grid(ls="--", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path = output_path(zone, f"fig_serie_{zone}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info(f"[{zone}] ✔ Graphique → {fig_path}")
    plt.close(fig)


# =============================================================================
# MAIN — process all configured zones
# =============================================================================

def process_zone(zone: str):
    bbox, zone_name = STUDY_AREAS[zone]
    log.info("─" * 60)
    log.info(f"  ZONE : {zone_name.upper()} ({zone})")
    log.info(f"  Bbox : {bbox}")
    log.info(f"  Period  : {WAR_START} → {WAR_END}")
    log.info("─" * 60)

    # 1. Contributions OSM
    gdf_contrib = fetch_contributions(zone, bbox)

    # 2. Suppressions
    gdf_del = fetch_deletions(zone, bbox)

    # 3. Ruines/destroyed
    gdf_ruins = fetch_ruins(zone, bbox)

    # 4. Bombardements ACLED (filtre event_type + sub_event_type)
    gdf_acled = load_acled_bombings(zone, bbox)

    # 5. Croisement OSM × ACLED
    gdf_match = gpd.GeoDataFrame()
    if not gdf_contrib.empty and not gdf_acled.empty:
        gdf_match = match_contributions_with_bombings(
            gdf_contrib, gdf_acled, zone
        )

    # 6. Monthly series
    df_series = build_monthly_series(zone, gdf_contrib, gdf_acled)

    # 7. Graphique
    if not df_series.empty:
        plot_series(zone, zone_name, df_series)

    # Summary
    log.info(f"[{zone}] ── SUMMARY ─────────────────────────────────")
    log.info(f"[{zone}] OSM contributions    : {len(gdf_contrib):>8,}")
    log.info(f"[{zone}] of which deletions   : {len(gdf_del):>8,}")
    log.info(f"[{zone}] Ruins/destroyed      : {len(gdf_ruins):>8,}")
    log.info(f"[{zone}] ACLED bombings       : {len(gdf_acled):>8,}")
    if not gdf_match.empty:
        field = f"has_bombing_{SPATIAL_BUFFER_M//1000}km_{WINDOW_DAYS}d"
        if field in gdf_match.columns:
            n_c = int(gdf_match[field].sum())
            pct = n_c / len(gdf_match) * 100
            log.info(f"[{zone}] Correlated contributions: "
                     f"{n_c:,}/{len(gdf_match):,} ({pct:.1f}%)")
    log.info(f"[{zone}] Files → {os.path.join(BASE_OUTPUT, zone)}/")


def main():
    log.info("═" * 60)
    log.info("  OSM UNDER THE TEST OF WAR")
    log.info(f"  Zones : {ZONES_TO_PROCESS}")
    log.info(f"  Period  : {WAR_START} → {WAR_END}")
    log.info("═" * 60)

    os.makedirs(BASE_OUTPUT, exist_ok=True)
    os.makedirs(BASE_CACHE,  exist_ok=True)

    for zone in ZONES_TO_PROCESS:
        if zone not in STUDY_AREAS:
            log.error(f"Unknown zone: {zone} — available zones: "
                      f"{list(STUDY_AREAS.keys())}")
            continue
        process_zone(zone)

    log.info("═" * 60)
    log.info("  DONE — all GeoJSON files are ready for QGIS")
    log.info("═" * 60)


if __name__ == "__main__":
    main()

