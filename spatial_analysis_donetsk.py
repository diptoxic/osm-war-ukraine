#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         OSM UNDER THE TEST OF WAR — Exploratory Analysis            ║
║              Area: Donetsk border (~50 km from the front)           ║
║              Period: Feb 2022 → Dec 2022                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Project [anonymized] / [anonymized]                           ║
║  Sponsor: Raphaël Bres                                              ║
╚══════════════════════════════════════════════════════════════════════╝

OSM signals analyzed:
  1. DELETED buildings (deleted=True in ohsome history)
  2. MODIFIED buildings with destruction tags
     (building=ruins, ruins=*, destroyed:building=*, disused:building=*)
  3. GLOBAL ACTIVITY DROP per spatial cell (H3-like grid)

Cross-references:
  - DeepStateMap front line (occupied zone over time)
  - ACLED events (air strikes, battles) in the 7 days
    preceding each OSM deletion/modification

Dependencies:
    pip install requests geopandas shapely matplotlib pandas numpy tqdm contextily
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from shapely.geometry import shape, Point, Polygon, box
from shapely import wkt
from tqdm import tqdm

try:
    import contextily as cx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("[INFO] contextily not available — maps without basemap")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================================
# ░░  CONFIGURATION  ░░
# =============================================================================

# ── Donetsk bounding box: border ~50 km around the 2022 front line ───────────
# Covers: Donetsk city, Mariupol, Volnovakha, Avdiivka, Sievierodonetsk, Sebastopol
BBOX = "36.8,47.2,39.2,49.0"   # lon_min, lat_min, lon_max, lat_max

# ── Period ────────────────────────────────────────────────────────────────────
START = "2022-02-01"
END   = "2022-12-31"

# ── Temporal granularity for aggregated counts ────────────────────────────────
INTERVAL = "P1M"   # monthly — reasonable for a bbox of this size

# ── OSM tags indicating destruction ──────────────────────────────────────────
# Ohsome filter: buildings with ruin/destruction tags
FILTER_DESTROYED = (
    "(building=ruins or ruins=yes or ruins=* "
    "or destroyed:building=* or disused:building=*) "
    "and type:way"
)
FILTER_BUILDINGS = "building=* and type:way"

# ── Files ─────────────────────────────────────────────────────────────────────
ACLED_FILE   = "dataACLED.shp"
OUTPUT_DIR   = "outputs"
DATA_DIR     = "data"

# ── ACLED time window before an OSM event (days) ─────────────────────────────
ACLED_WINDOW_DAYS = 7

# ── Metric CRS for distance calculations (UTM 37N — Donetsk) ─────────────────
UTM_CRS = "EPSG:32637"

# ── Activity grid resolution (degrees) ───────────────────────────────────────
GRID_RES = 0.05   # ~5 km — fine enough without blowing up memory


# =============================================================================
# ░░  1. FRONT LINE — Historical data cyterat/deepstate-map-data  ░░
# =============================================================================

# URL of the consolidated historical file (updated daily)
DEEPSTATE_HISTORICAL_URL = (
    "https://raw.githubusercontent.com/cyterat/deepstate-map-data"
    "/main/deepstate-map-data.geojson.gz"
)
DEEPSTATE_LOCAL_GZ   = os.path.join(DATA_DIR, "deepstate-map-data.geojson.gz")
DEEPSTATE_LOCAL_JSON = os.path.join(DATA_DIR, "deepstate-map-data.geojson")


def fetch_frontline(target_date: str = None, data_dir: str = DATA_DIR) -> gpd.GeoDataFrame:
    """
    Loads the DeepStateMap front line at a specific date.

    Source: consolidated historical GeoJSON file from the cyterat/deepstate-map-data repo.
    Each feature contains a 'date' column (YYYY-MM-DD) corresponding
    to the daily snapshot of the occupied zone.

    If the file does not exist locally, it is downloaded (once only).
    If target_date is None, uses END (end of the analysis period).

    Returns an EPSG:4326 GeoDataFrame with the occupied zone geometry
    at the snapshot closest to target_date.
    """
    if target_date is None:
        target_date = END
    os.makedirs(data_dir, exist_ok=True)

    # ── Download if absent (file ~100 MB, once only) ──────────────────────────
    if not os.path.exists(DEEPSTATE_LOCAL_GZ):
        log.info(f"Downloading DeepState historical file (~100 MB)…")
        log.info(f"  Source: {DEEPSTATE_HISTORICAL_URL}")
        try:
            r = requests.get(DEEPSTATE_HISTORICAL_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(DEEPSTATE_LOCAL_GZ, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            log.info(f"  File downloaded → {DEEPSTATE_LOCAL_GZ}")
        except requests.RequestException as e:
            log.error(f"Download failed: {e}")
            log.error(
                "Download the file manually from:\n"
                "  https://github.com/cyterat/deepstate-map-data\n"
                f"  and place it at: {DEEPSTATE_LOCAL_GZ}"
            )
            sys.exit(1)

    # ── Read the historical file ───────────────────────────────────────────────
    log.info(f"Loading DeepState historical file…")
    try:
        import gzip as gz
        with gz.open(DEEPSTATE_LOCAL_GZ, "rt", encoding="utf-8") as f:
            all_gdf = gpd.read_file(f)
    except Exception:
        # Try without decompression (file already decompressed)
        try:
            all_gdf = gpd.read_file(DEEPSTATE_LOCAL_GZ)
        except Exception as e:
            log.error(f"Cannot read DeepState file: {e}")
            sys.exit(1)

    # Normalize the date column
    date_col = next(
        (c for c in all_gdf.columns if "date" in c.lower()), None
    )
    if date_col is None:
        log.error("Column 'date' not found in the DeepState file.")
        sys.exit(1)

    all_gdf["_date"] = pd.to_datetime(all_gdf[date_col], errors="coerce", utc=True).dt.tz_convert(None)

    # ── Select the snapshot closest to target_date ────────────────────────────
    target_ts  = pd.Timestamp(target_date)
    available  = all_gdf["_date"].dropna().unique()
    available  = pd.DatetimeIndex(sorted(available))

    # Find the available snapshot ≤ target_date
    before = available[available <= target_ts]
    if len(before) == 0:
        log.warning(f"No snapshot before {target_date} — using the oldest available.")
        chosen = available[0]
    else:
        chosen = before[-1]

    log.info(f"Front line — selected snapshot: {chosen.date()} (target: {target_date})")

    front_gdf = all_gdf[all_gdf["_date"] == chosen].copy()

    if front_gdf.empty:
        log.error(f"No geometry for snapshot {chosen.date()}")
        sys.exit(1)

    if front_gdf.crs is None:
        front_gdf = front_gdf.set_crs(4326)
    elif front_gdf.crs.to_epsg() != 4326:
        front_gdf = front_gdf.to_crs(4326)

    # Save the used snapshot
    cache_path = os.path.join(data_dir, f"frontline_{chosen.date()}.geojson")
    front_gdf.to_file(cache_path, driver="GeoJSON")
    log.info(f"Snapshot saved → {cache_path}")

    return front_gdf


# =============================================================================
# ░░  2. ACLED DATA  ░░
# =============================================================================

def load_acled(filepath: str, bbox_str: str = BBOX) -> gpd.GeoDataFrame:
    """
    Loads dataACLED.shp, normalizes columns, filters on bbox and
    analysis period.
    """
    log.info(f"Loading ACLED: {filepath}")
    gdf = gpd.read_file(filepath)
    gdf.columns = [c.lower() for c in gdf.columns]

    # CRS normalization
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Detect and parse the date column
    date_col = next(
        (c for c in gdf.columns if "date" in c or "timestamp" in c), None
    )
    if date_col:
        raw_dates    = pd.to_datetime(gdf[date_col], errors="coerce", utc=True)
        gdf["date"]  = raw_dates.dt.tz_localize(None) if raw_dates.dt.tz is None \
                       else raw_dates.dt.tz_convert(None)
        log.info(f"Dates parsed — example: {gdf['date'].dropna().iloc[0] if not gdf['date'].dropna().empty else 'N/A'}")
    else:
        log.warning("Date column not found in ACLED.")

    # Filter Ukraine if country column is available
    for col in ["country", "pays", "adm0_name", "admin0"]:
        if col in gdf.columns:
            gdf = gdf[gdf[col].str.contains("Ukraine", case=False, na=False)]
            log.info(f"Country filter via '{col}' → {len(gdf)} Ukraine events")
            break

    # Bbox filter
    lon_min, lat_min, lon_max, lat_max = map(float, bbox_str.split(","))
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max]

    # Period filter
    if "date" in gdf.columns:
        gdf = gdf[
            (gdf["date"] >= pd.Timestamp(START)) &
            (gdf["date"] <= pd.Timestamp(END))
        ]

    log.info(f"ACLED filtered: {len(gdf)} events in the area/period")
    return gdf.reset_index(drop=True)


# =============================================================================
# ░░  3. OHSOME QUERIES  ░░
# =============================================================================

def _ohsome_post(endpoint: str, params: dict) -> dict:
    """
    Performs a POST request to the ohsome API with retry.
    """
    URL = f"https://api.ohsome.org/v1/{endpoint}"
    for attempt in range(3):
        try:
            r = requests.post(URL, data=params, timeout=620)
            if r.status_code == 200:
                return r.json()
            log.warning(f"ohsome HTTP {r.status_code} (attempt {attempt+1}/3): {r.text[:200]}")
        except requests.RequestException as e:
            log.warning(f"ohsome network error (attempt {attempt+1}/3): {e}")
        time.sleep(5)
    log.error(f"ohsome endpoint '{endpoint}' unreachable after 3 attempts.")
    return {}


def fetch_osm_deletions(bbox: str, start: str, end: str) -> pd.DataFrame:
    """
    Retrieves via ohsome /contributions/count the number of DELETION
    contributions on buildings, per month.

    The contributions/count endpoint with contributionType=deletion
    returns the number of OSM object deletions in the period.
    """
    log.info("Ohsome query — building deletions (monthly)…")
    params = {
        "bboxes":           bbox,
        "time":             f"{start}/{end}/{INTERVAL}",
        "filter":           FILTER_BUILDINGS,
        "contributionType": "deletion",
        "timeout":          "600"
    }
    data = _ohsome_post("contributions/count", params)
    rows = data.get("result", [])
    if not rows:
        log.warning("No deletion data received.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period"]    = pd.to_datetime(df["fromTimestamp"])
    df["deletions"] = df["value"].fillna(0).astype(int)
    log.info(f"Deletions received: {df['deletions'].sum()} over {len(df)} periods")
    return df[["period", "deletions"]]


def fetch_osm_destroyed_tags(bbox: str, start: str, end: str) -> pd.DataFrame:
    """
    Counts buildings with destruction tags (ruins, destroyed…)
    per month. Uses /elements/count to get the state at each snapshot.
    """
    log.info("Ohsome query — buildings with destruction tags (monthly)…")
    params = {
        "bboxes":  bbox,
        "time":    f"{start}/{end}/{INTERVAL}",
        "filter":  FILTER_DESTROYED,
        "timeout": "600"
    }
    data = _ohsome_post("elements/count", params)
    rows = data.get("result", [])
    if not rows:
        log.warning("No destruction tag data received.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period"]   = pd.to_datetime(df["timestamp"])
    df["n_ruines"] = df["value"].fillna(0).astype(int)
    log.info(f"Ruins snapshot — monthly max: {df['n_ruines'].max()}")
    return df[["period", "n_ruines"]]


def fetch_osm_activity(bbox: str, start: str, end: str) -> pd.DataFrame:
    """
    Counts all OSM contributions (buildings) per month to
    measure the global activity drop in the area.
    """
    log.info("Ohsome query — global OSM building activity (monthly)…")
    params = {
        "bboxes":  bbox,
        "time":    f"{start}/{end}/{INTERVAL}",
        "filter":  FILTER_BUILDINGS,
        "timeout": "600"
    }
    data = _ohsome_post("contributions/count", params)
    rows = data.get("result", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period"]   = pd.to_datetime(df["fromTimestamp"])
    df["activity"] = df["value"].fillna(0).astype(int)
    return df[["period", "activity"]]


def fetch_osm_deletions_geom(bbox: str, start: str, end: str) -> gpd.GeoDataFrame:
    """
    Retrieves centroids of deleted buildings via
    /contributions/centroid — lightweight endpoint that returns ONLY
    one point per contribution (not the full geometry), avoiding
    any MemoryError even on a large area.

    contributionType=deletion filters server-side: only actually
    deleted objects are returned.

    If this endpoint fails (400), falls back to a counting grid
    per cell via /contributions/count with groupByBbox.
    """
    log.info("Ohsome query — deletion centroids (contributions/centroid)…")
    params = {
        "bboxes":           bbox,
        "time":             f"{start}/{end}",
        "filter":           FILTER_BUILDINGS,
        "contributionType": "deletion",
        "properties":       "metadata",
        "timeout":          "600"
    }

    URL  = "https://api.ohsome.org/v1/contributions/centroid"
    try:
        r = requests.post(URL, data=params, timeout=620)
        if r.status_code == 200:
            data     = r.json()
            features = data.get("features", [])
            if features:
                log.info(f"{len(features)} deletion centroids received")
                pts, timestamps = [], []
                for f in features:
                    try:
                        geom = shape(f["geometry"])   # déjà un Point
                        pts.append(geom)
                        ts = f.get("properties", {}).get(
                            "@toTimestamp",
                            f.get("properties", {}).get("@snapshotTimestamp", end)
                        )
                        timestamps.append(pd.to_datetime(ts, errors="coerce"))
                    except Exception:
                        continue
                gdf = gpd.GeoDataFrame({"deleted_at": timestamps}, geometry=pts, crs=4326)
                log.info(f"Deletions GeoDataFrame: {len(gdf)} points")
                return gdf
            else:
                log.warning("contributions/centroid: no features returned.")
        else:
            log.warning(
                f"contributions/centroid HTTP {r.status_code}: {r.text[:200]}\n"
                "→ Falling back to counting grid…"
            )
    except (requests.RequestException, MemoryError) as e:
        log.warning(f"contributions/centroid error: {e} → Falling back to grid…")

    # ── Fallback: per-cell counting grid via groupByBbox ─────────────────────
    return _deletions_grid_fallback(bbox, start, end)


def _deletions_grid_fallback(bbox: str, start: str, end: str) -> gpd.GeoDataFrame:
    """
    Fallback when contributions/centroid is not available.

    Splits the bbox into GRID_RES×GRID_RES cells and queries
    contributions/count (deletion) for each cell.
    Returns a GeoDataFrame with one point at the centroid of each
    cell, repeated as many times as there were deletions.
    → Allows feeding build_activity_grid() with the same data.

    Much lighter than elementsFullHistory since only integers are
    retrieved, never full geometries.
    """
    log.info("Fallback: counting deletions per grid cell…")
    lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))

    lons = np.arange(lon_min, lon_max, GRID_RES)
    lats = np.arange(lat_min, lat_max, GRID_RES)
    n_cells = len(lons) * len(lats)
    log.info(f"  {n_cells} cells to query (resolution {GRID_RES}°)…")

    all_pts, all_ts = [], []
    URL = "https://api.ohsome.org/v1/contributions/count"

    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            cell_bbox = (
                f"{lon:.4f},{lat:.4f},"
                f"{min(lon+GRID_RES, lon_max):.4f},"
                f"{min(lat+GRID_RES, lat_max):.4f}"
            )
            params = {
                "bboxes":           cell_bbox,
                "time":             f"{start}/{end}",
                "filter":           FILTER_BUILDINGS,
                "contributionType": "deletion",
                "timeout":          "60"
            }
            try:
                r = requests.post(URL, data=params, timeout=70)
                if r.status_code != 200:
                    continue
                rows = r.json().get("result", [])
                n_del = sum(row.get("value", 0) for row in rows)
                if n_del > 0:
                    # cell centroid, repeated n_del times
                    cx_pt = lon + GRID_RES / 2
                    cy_pt = lat + GRID_RES / 2
                    for _ in range(int(n_del)):
                        all_pts.append(Point(cx_pt, cy_pt))
                        all_ts.append(pd.to_datetime(end))
            except Exception:
                continue

        # Log progress every 5 columns
        if (i + 1) % 5 == 0:
            log.info(f"  Progress: column {i+1}/{len(lons)} done")

    if not all_pts:
        log.warning("Grid fallback: no deletions detected.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame({"deleted_at": all_ts}, geometry=all_pts, crs=4326)
    log.info(f"Grid fallback — {len(gdf)} deletions reconstructed")
    return gdf


# =============================================================================
# ░░  4. OSM × ACLED CROSS-CORRELATION  ░░
# =============================================================================

def _strip_tz(series: pd.Series) -> pd.Series:
    """
    Normalizes a datetime Series: removes the timezone regardless of
    its origin (UTC, tz-aware, tz-naive). Always returns tz-naive.
    """
    try:
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    try:
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def correlate_osm_acled(
    df_osm: pd.DataFrame,
    gdf_acled: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    For each month, counts the number of ACLED events that occurred in
    the ACLED_WINDOW_DAYS days preceding the end of the OSM period.
    Returns df_osm enriched with 'n_acled_events' and 'n_acled_fatalities'.
    """
    if "date" not in gdf_acled.columns or df_osm.empty:
        return df_osm

    # ── Guarantee: ACLED dates always tz-naive ────────────────────────────
    acled_dates = _strip_tz(gdf_acled["date"])

    # ── Guarantee: OSM period always tz-naive ─────────────────────────────
    osm_periods = _strip_tz(pd.to_datetime(df_osm["period"], errors="coerce"))

    acled_counts, acled_fatal = [], []
    fatal_col = next(
        (c for c in gdf_acled.columns if "fatal" in c or "death" in c), None
    )

    for period_end in osm_periods:
        if pd.isna(period_end):
            acled_counts.append(0)
            acled_fatal.append(0)
            continue
        window_start = period_end - timedelta(days=ACLED_WINDOW_DAYS)
        mask = (acled_dates >= window_start) & (acled_dates <= period_end)
        subset = gdf_acled[mask]
        acled_counts.append(len(subset))

        if fatal_col:
            acled_fatal.append(pd.to_numeric(subset[fatal_col], errors="coerce").sum())
        else:
            acled_fatal.append(0)

    df_osm = df_osm.copy()
    df_osm["n_acled_events"]     = acled_counts
    df_osm["n_acled_fatalities"] = acled_fatal
    return df_osm


# =============================================================================
# ░░  5. SPATIAL ACTIVITY GRID  ░░
# =============================================================================

def build_activity_grid(
    gdf_deletions: gpd.GeoDataFrame,
    bbox_str: str = BBOX,
    resolution: float = GRID_RES
) -> gpd.GeoDataFrame:
    """
    Builds a regular grid over the bbox and counts deletions
    per cell → destruction heat map.
    """
    if gdf_deletions.empty:
        log.warning("No deletion data for the grid.")
        return gpd.GeoDataFrame()

    lon_min, lat_min, lon_max, lat_max = map(float, bbox_str.split(","))

    lons = np.arange(lon_min, lon_max, resolution)
    lats = np.arange(lat_min, lat_max, resolution)

    cells, counts = [], []
    pts = np.array([[g.x, g.y] for g in gdf_deletions.geometry])

    for lon in lons:
        for lat in lats:
            cell = box(lon, lat, lon + resolution, lat + resolution)
            n = np.sum(
                (pts[:, 0] >= lon) & (pts[:, 0] < lon + resolution) &
                (pts[:, 1] >= lat) & (pts[:, 1] < lat + resolution)
            )
            if n > 0:
                cells.append(cell)
                counts.append(int(n))

    if not cells:
        return gpd.GeoDataFrame()

    grid = gpd.GeoDataFrame({"n_deletions": counts}, geometry=cells, crs=4326)
    log.info(f"Grid: {len(grid)} cells with at least 1 deletion")
    return grid


# =============================================================================
# ░░  6. VISUALIZATIONS  ░░
# =============================================================================

def plot_all(
    df_del:     pd.DataFrame,
    df_ruins:   pd.DataFrame,
    df_act:     pd.DataFrame,
    df_corr:    pd.DataFrame,
    gdf_del:    gpd.GeoDataFrame,
    gdf_acled:  gpd.GeoDataFrame,
    front_gdf:  gpd.GeoDataFrame,
    grid:       gpd.GeoDataFrame,
    output_dir: str = OUTPUT_DIR
):
    os.makedirs(output_dir, exist_ok=True)
    C = {
        "deletion": "#E74C3C",
        "ruins":    "#E67E22",
        "activity": "#3498DB",
        "acled":    "#8E44AD",
        "front":    "#27AE60",
        "gray":     "#95A5A6",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 1 — OSM temporal dashboard (3 signals on 3 rows)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Signaux OSM — Frontière Donetsk  |  Fév–Déc 2022",
        fontsize=15, fontweight="bold", y=0.99
    )

    if not df_act.empty:
        axes[0].bar(df_act["period"], df_act["activity"],
                    color=C["activity"], alpha=0.85, width=25)
        axes[0].set_ylabel("Contributions / mois", fontsize=10)
        axes[0].set_title("① Activité globale des contributeurs OSM (bâtiments)", fontsize=11)
        axes[0].axvline(pd.Timestamp("2022-02-24"), color="red",
                        linestyle="--", lw=1.5, label="Invasion 24 fév.")
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    if not df_del.empty:
        axes[1].bar(df_del["period"], df_del["deletions"],
                    color=C["deletion"], alpha=0.85, width=25)
        axes[1].set_ylabel("Suppressions / mois", fontsize=10)
        axes[1].set_title("② Bâtiments OSM supprimés (signal de destruction physique)", fontsize=11)
        axes[1].axvline(pd.Timestamp("2022-02-24"), color="red",
                        linestyle="--", lw=1.5)
        # Peak annotation
        if len(df_del) > 0:
            peak = df_del.loc[df_del["deletions"].idxmax()]
            axes[1].annotate(
                f"Pic : {int(peak['deletions'])} suppressions",
                xy=(peak["period"], peak["deletions"]),
                xytext=(peak["period"], peak["deletions"] * 1.12),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    if not df_ruins.empty:
        axes[2].plot(df_ruins["period"], df_ruins["n_ruines"],
                     color=C["ruins"], marker="o", lw=2.5, ms=7,
                     label="Bâtiments tagués ruins/destroyed")
        axes[2].fill_between(df_ruins["period"], df_ruins["n_ruines"],
                              alpha=0.18, color=C["ruins"])
        axes[2].set_ylabel("Nbre de bâtiments", fontsize=10)
        axes[2].set_title("③ Bâtiments OSM portant des tags ruins / destroyed / disused", fontsize=11)
        axes[2].axvline(pd.Timestamp("2022-02-24"), color="red",
                        linestyle="--", lw=1.5)
        axes[2].legend(fontsize=9)
        axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=40, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(output_dir, "fig1_signaux_osm.png"), dpi=150, bbox_inches="tight")
    log.info("✔ Fig 1 — OSM signals saved")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 2 — OSM deletions vs ACLED comparison (dual axis)
    # ══════════════════════════════════════════════════════════════════════════
    if not df_corr.empty and "n_acled_events" in df_corr.columns:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        bars = ax1.bar(df_corr["period"], df_corr["deletions"],
                       color=C["deletion"], alpha=0.7, width=22,
                       label="Suppressions OSM / mois")
        ax2.plot(df_corr["period"], df_corr["n_acled_events"],
                 color=C["acled"], marker="D", lw=2.2, ms=8,
                 label=f"Événements ACLED (fenêtre J-{ACLED_WINDOW_DAYS})")
        ax2.fill_between(df_corr["period"], df_corr["n_acled_events"],
                         alpha=0.12, color=C["acled"])

        ax1.axvline(pd.Timestamp("2022-02-24"), color="red",
                    linestyle="--", lw=1.5, label="Invasion 24 fév.")
        ax1.set_ylabel("Suppressions OSM / mois", color=C["deletion"], fontsize=11, labelpad=10)
        ax2.set_ylabel(f"Événements ACLED\n(fenêtre {ACLED_WINDOW_DAYS}j avant fin de mois)",
                       color=C["acled"], fontsize=11, labelpad=10)
        ax1.set_title(
            "Comparaison : suppressions OSM vs intensité des combats ACLED\n"
            "Zone Donetsk — Fév–Déc 2022",
            fontsize=13, fontweight="bold"
        )
        # Merged legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper left",
                   framealpha=0.9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)

        # Pearson correlation coefficient
        if len(df_corr) > 2:
            corr_val = df_corr["deletions"].corr(df_corr["n_acled_events"])
            ax1.text(0.98, 0.95,
                     f"r de Pearson = {corr_val:.2f}",
                     transform=ax1.transAxes, ha="right", va="top",
                     fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                               ec=C["acled"], alpha=0.9))

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig2_osm_vs_acled.png"), dpi=150, bbox_inches="tight")
        log.info("✔ Fig 2 — OSM vs ACLED saved")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 3 — Map: deletion grid + front line + ACLED
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 10))
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    if not front_gdf.empty:
        front_gdf.plot(ax=ax, color=C["front"], alpha=0.13, zorder=2)
        front_gdf.boundary.plot(ax=ax, color=C["front"], lw=2.2,
                                label="Zone occupée (DeepState, actuelle)", zorder=3)

    if not grid.empty:
        vmax = max(grid["n_deletions"].quantile(0.97), 1)
        grid.plot(column="n_deletions", ax=ax,
                  cmap="YlOrRd", alpha=0.78,
                  norm=Normalize(vmin=0, vmax=vmax),
                  legend=True,
                  legend_kwds={"label": "Suppressions OSM / cellule (~5 km)",
                               "shrink": 0.55, "orientation": "vertical"},
                  zorder=4)

    if not gdf_acled.empty:
        # Point size proportional to number of fatalities if available
        fatal_col = next(
            (c for c in gdf_acled.columns if "fatal" in c or "death" in c), None
        )
        if fatal_col:
            sizes = np.clip(
                pd.to_numeric(gdf_acled[fatal_col], errors="coerce").fillna(1) * 1.5,
                2, 40
            )
        else:
            sizes = 5
        gdf_acled.plot(ax=ax, color=C["acled"], markersize=sizes,
                       alpha=0.45, label="Événements ACLED", zorder=5)

    if HAS_CTX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.DarkMatter, zoom=9)
        except Exception as e:
            log.warning(f"Basemap not loaded: {e}")

    ax.set_title(
        "Carte des destructions OSM — Frontière Donetsk, Fév–Déc 2022\n"
        "Grille de suppressions OSM  +  Événements ACLED  +  Ligne de front DeepStateMap",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.88)
    ax.grid(linestyle="--", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_carte_synthese.png"), dpi=150, bbox_inches="tight")
    log.info("✔ Fig 3 — synthesis map saved")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 4 — ACLED event types (horizontal bars)
    # ══════════════════════════════════════════════════════════════════════════
    type_col = next(
        (c for c in gdf_acled.columns if "event" in c and "type" in c), None
    ) if not gdf_acled.empty else None

    if type_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Analyse des événements ACLED — Zone Donetsk, Fév–Déc 2022",
                     fontsize=13, fontweight="bold")

        # Left: event types
        counts = gdf_acled[type_col].value_counts().head(8)
        colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        counts.plot(kind="barh", ax=axes[0], color=colors, edgecolor="white")
        axes[0].set_title("Types d'événements", fontsize=11)
        axes[0].set_xlabel("Nombre d'événements")
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)

        # Right: monthly evolution by type (top 4)
        if "date" in gdf_acled.columns:
            top4 = counts.head(4).index.tolist()
            palette = plt.cm.Set1(np.linspace(0, 0.8, 4))
            for k, (typ, col) in enumerate(zip(top4, palette)):
                sub = gdf_acled[gdf_acled[type_col] == typ].copy()
                monthly = (
                    sub.groupby(sub["date"].dt.to_period("M"))
                    .size().reset_index(name="n")
                )
                monthly["date"] = monthly["date"].dt.to_timestamp()
                axes[1].plot(monthly["date"], monthly["n"],
                             marker="o", lw=1.8, ms=5,
                             label=typ[:30], color=col, alpha=0.85)
            axes[1].set_title("Évolution mensuelle (top 4 types)", fontsize=11)
            axes[1].set_ylabel("Nombre d'événements")
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=40, ha="right")
            axes[1].legend(fontsize=8, loc="upper right")
            axes[1].grid(linestyle="--", alpha=0.35)
            axes[1].axvline(pd.Timestamp("2022-02-24"), color="red",
                            linestyle="--", lw=1.2, label="Invasion")

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_acled_types.png"), dpi=150, bbox_inches="tight")
        log.info("✔ Fig 4 — ACLED types saved")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 5 — Triple comparison: OSM activity + deletions + ACLED
    #          on the same timeline with annotated zones
    # ══════════════════════════════════════════════════════════════════════════
    if not df_del.empty and not df_act.empty:
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 65))

        # Global activity (gray background)
        if not df_act.empty:
            ax1.bar(df_act["period"], df_act["activity"],
                    color=C["gray"], alpha=0.45, width=22,
                    label="Activité OSM totale", zorder=1)

        # Deletions (red)
        ax1.bar(df_del["period"], df_del["deletions"],
                color=C["deletion"], alpha=0.8, width=15,
                label="Suppressions OSM", zorder=2)

        # Ruin tags (orange)
        if not df_ruins.empty:
            ax2.plot(df_ruins["period"], df_ruins["n_ruines"],
                     color=C["ruins"], lw=2, ms=6, marker="s",
                     label="Tags ruins/destroyed", zorder=3)

        # Monthly ACLED
        if not gdf_acled.empty and "date" in gdf_acled.columns:
            monthly_acled = (
                gdf_acled
                .groupby(gdf_acled["date"].dt.to_period("M"))
                .size().reset_index(name="count")
            )
            monthly_acled["date"] = monthly_acled["date"].dt.to_timestamp()
            ax3.plot(monthly_acled["date"], monthly_acled["count"],
                     color=C["acled"], lw=2, ms=7, marker="D",
                     linestyle="--", label="Événements ACLED", zorder=4)

        # Invasion line
        ax1.axvline(pd.Timestamp("2022-02-24"), color="red",
                    linestyle="-", lw=2, alpha=0.7, label="Invasion 24 fév. 2022")

        # War phase zones
        phases = [
            ("2022-02-24", "2022-04-01", "#E74C3C", "Phase offensive\ninitiiale"),
            ("2022-04-01", "2022-09-01", "#E67E22", "Guerre\nd'attrition"),
            ("2022-09-01", "2022-12-31", "#27AE60", "Contre-\noffensives UKR"),
        ]
        for p_start, p_end, color, label in phases:
            ax1.axvspan(pd.Timestamp(p_start), pd.Timestamp(p_end),
                        alpha=0.06, color=color)
            ax1.text(
                pd.Timestamp(p_start) + (pd.Timestamp(p_end) - pd.Timestamp(p_start)) / 2,
                ax1.get_ylim()[1] * 0.88, label,
                ha="center", va="top", fontsize=8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7)
            )

        ax1.set_ylabel("Contributions / Suppressions OSM", fontsize=10)
        ax2.set_ylabel("Bâtiments tags ruins", color=C["ruins"], fontsize=10)
        ax3.set_ylabel("Événements ACLED / mois", color=C["acled"], fontsize=10)
        ax1.set_title(
            "Synthèse : activité OSM, destructions et combats ACLED\n"
            "Frontière Donetsk — Fév–Déc 2022",
            fontsize=13, fontweight="bold"
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")

        # Merged legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=9,
                   loc="upper left", framealpha=0.92, ncol=2)
        ax1.grid(linestyle="--", alpha=0.25)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig5_synthese_triple.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 5 — triple synthesis saved")
        plt.close(fig)

    log.info(f"All figures saved in '{output_dir}/'")
    log.info("Output files:")
    for f in sorted(os.listdir(output_dir)):
        log.info(f"  {output_dir}/{f}")

# =============================================================================
# ░░  MAIN  ░░
# =============================================================================

def main():
    log.info("═" * 60)
    log.info("  OSM UNDER THE TEST OF WAR — Exploratory Analysis")
    log.info(f"  Area   : {BBOX}")
    log.info(f"  Period : {START} → {END}")
    log.info("═" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ── 1. Front line at the END date of the period ───────────────────────────
    # Load the DeepState snapshot closest to END (2022-12-31)
    # to get the front line contemporary with the OSM data analyzed.
    front_gdf = fetch_frontline(target_date=END)

    # ── 2. ACLED ──────────────────────────────────────────────────────────────
    if os.path.exists(ACLED_FILE):
        gdf_acled = load_acled(ACLED_FILE)
        gdf_acled.to_file(os.path.join(OUTPUT_DIR, "acled_zone.geojson"), driver="GeoJSON")
        log.info(f"✔ ACLED saved ({len(gdf_acled)} events)")
    else:
        log.warning(f"{ACLED_FILE} not found — ACLED analysis disabled.")
        gdf_acled = gpd.GeoDataFrame()

    # ── 3. Ohsome queries — immediate save after each call ────────────────────
    df_deletions = fetch_osm_deletions(BBOX, START, END)
    if not df_deletions.empty:
        df_deletions.to_csv(os.path.join(OUTPUT_DIR, "deletions_mensuelles.csv"), index=False)
        log.info("✔ deletions_mensuelles.csv saved")

    df_ruins = fetch_osm_destroyed_tags(BBOX, START, END)
    if not df_ruins.empty:
        df_ruins.to_csv(os.path.join(OUTPUT_DIR, "ruins_mensuels.csv"), index=False)
        log.info("✔ ruins_mensuels.csv saved")

    df_activity = fetch_osm_activity(BBOX, START, END)
    if not df_activity.empty:
        df_activity.to_csv(os.path.join(OUTPUT_DIR, "activite_osm.csv"), index=False)
        log.info("✔ activite_osm.csv saved")

    log.info("Deleted geometry query (contributions/centroid or grid fallback)…")
    gdf_deletions_geom = fetch_osm_deletions_geom(BBOX, START, END)
    if not gdf_deletions_geom.empty:
        gdf_deletions_geom.to_file(
            os.path.join(OUTPUT_DIR, "suppressions_geom.geojson"), driver="GeoJSON"
        )
        log.info("✔ suppressions_geom.geojson saved")

    # ── 4. OSM × ACLED cross-correlation ─────────────────────────────────────
    df_corr = correlate_osm_acled(df_deletions, gdf_acled)
    if not df_corr.empty:
        df_corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_osm_acled.csv"), index=False)
        log.info("✔ correlation_osm_acled.csv saved")

    # ── 5. Spatial grid ───────────────────────────────────────────────────────
    grid = build_activity_grid(gdf_deletions_geom)
    if not grid.empty:
        grid.to_file(os.path.join(OUTPUT_DIR, "grille_suppressions.geojson"), driver="GeoJSON")
        log.info("✔ grille_suppressions.geojson saved")

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    plot_all(
        df_del    = df_deletions,
        df_ruins  = df_ruins,
        df_act    = df_activity,
        df_corr   = df_corr,
        gdf_del   = gdf_deletions_geom,
        gdf_acled = gdf_acled,
        front_gdf = front_gdf,
        grid      = grid
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  FINAL SUMMARY")
    log.info(f"  Total OSM deletions       : {df_deletions['deletions'].sum() if not df_deletions.empty else 'N/A'}")
    log.info(f"  Ruins buildings max/month : {df_ruins['n_ruines'].max() if not df_ruins.empty else 'N/A'}")
    log.info(f"  ACLED events retained     : {len(gdf_acled)}")
    log.info(f"  Geolocated deletions      : {len(gdf_deletions_geom) if not gdf_deletions_geom.empty else 0}")
    log.info(f"  Output files              : {OUTPUT_DIR}/")
    log.info("═" * 60)


if __name__ == "__main__":
    main()

