#!/usr/bin/env python3
"""
Spatio-temporal analysis: ACLED bombings × OSM edits (Kyiv, 2022)
Hypothesis: an OSM edit is a "response" to a bombing if it is:
  - within RADIUS_M meters of the bombing location
  - between 0 and 7 days AFTER the bombing date

Output:
  - acled_with_osm_response.geojson  → each bombing with count of OSM responses
  - matched_pairs.csv                → every bombing–edit pair
  - fig_kyiv_response.png            → map + time chart for the presentation

Usage:
    pip install requests geopandas shapely pandas matplotlib contextily
    python spatiotemporal_kyiv.py
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from shapely.geometry import Point

try:
    import contextily as cx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
RADIUS_M    = 500        # spatial radius around each bombing (metres)
LAG_DAYS    = 7          # max days after bombing to count as a response
UTM_CRS     = "EPSG:32637"  # UTM for Ukraine (metres)

OUTPUT_DIR  = "outputs_kyiv_1y"
ACLED_FILE  = "data/dataACLED_Kyiv.shp"          # your ACLED shapefile
OSM_FILE    = os.path.join(OUTPUT_DIR, "kyiv_osm_edits_2022.geojson")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_acled_kyiv():
    gdf = gpd.read_file(ACLED_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    date_col = next((c for c in gdf.columns if "date" in c or "timestamp" in c), None)
    if date_col:
        gdf["bomb_date"] = pd.to_datetime(
            gdf[date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)

    # Ukraine only
    for col in ["country", "pays", "adm0_name", "admin0"]:
        if col in gdf.columns:
            gdf = gdf[gdf[col].str.contains("Ukraine", case=False, na=False)]
            break

    # Kyiv Oblast
    for col in ["admin1", "region", "oblast", "adm1_name"]:
        if col in gdf.columns:
            mask = gdf[col].str.contains("Kyiv|Kiev|Київ", case=False, na=False)
            if mask.sum() > 0:
                gdf = gdf[mask]
                print(f"  Kyiv filter on '{col}': {len(gdf)} events")
                break

    # Time window
    gdf = gdf[
        (gdf["bomb_date"] >= pd.Timestamp("2022-02-24")) &
        (gdf["bomb_date"] <= pd.Timestamp("2022-12-31"))
    ]

    # ── FIX 1: keep only high geo-precision events ─────────────────────────
    # ACLED geo_precis: 1 = exact location, 2 = near town, 3+ = centroid/admin
    if "geo_precis" in gdf.columns:
        before = len(gdf)
        gdf = gdf[pd.to_numeric(gdf["geo_precis"], errors="coerce") <= 2]
        print(f"  Geo-precision filter: {before} → {len(gdf)} events kept")

    # ── FIX 2: deduplicate events at identical coordinates (same day + same point)
    gdf["_lon"] = gdf.geometry.x.round(5)
    gdf["_lat"] = gdf.geometry.y.round(5)
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["_lon", "_lat", "bomb_date"])
    gdf = gdf.drop(columns=["_lon", "_lat"])
    print(f"  Dedup filter: {before} → {len(gdf)} events kept")

    # Explosions / shelling only
    for col in ["event_type", "type", "event"]:
        if col in gdf.columns:
            mask = gdf[col].str.contains(
                "Explosion|Shelling|Airstrike|Bombing|Air", case=False, na=False
            )
            if mask.sum() > 0:
                gdf = gdf[mask]
                print(f"  Explosion filter on '{col}': {len(gdf)} events")
            break

    print(f"ACLED Kyiv final: {len(gdf)} bombing events")
    return gdf.reset_index(drop=True)


def load_osm_edits():
    """Load OSM building edits produced by fetch_kyiv_osm_edits.py"""
    gdf = gpd.read_file(OSM_FILE)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf["edit_date"] = pd.to_datetime(gdf["timestamp"], errors="coerce")
    print(f"OSM edits loaded: {len(gdf)} contributions")
    return gdf.reset_index(drop=True)


# ── SPATIO-TEMPORAL JOIN ───────────────────────────────────────────────────────
def spatiotemporal_join(gdf_acled, gdf_osm):
    """
    For each bombing, find OSM edits within RADIUS_M metres
    and within LAG_DAYS days after the bombing.

    Returns:
        gdf_acled_enriched  — bombings with n_osm_response column
        df_pairs            — all matched pairs
    """
    print(f"\nRunning spatio-temporal join (radius={RADIUS_M}m, lag={LAG_DAYS}d)...")

    # Reproject to UTM for metre-based buffer
    acled_utm = gdf_acled.to_crs(UTM_CRS).copy()
    osm_utm   = gdf_osm.to_crs(UTM_CRS).copy()

    pairs = []

    for idx, bomb in acled_utm.iterrows():
        if pd.isna(bomb.geometry) or pd.isna(bomb["bomb_date"]):
            continue

        # Spatial filter: within RADIUS_M metres
        buf       = bomb.geometry.buffer(RADIUS_M)
        nearby    = osm_utm[osm_utm.geometry.within(buf)].copy()

        if nearby.empty:
            continue

        # Temporal filter: 0–LAG_DAYS days after the bombing
        nearby["delta_days"] = (
            nearby["edit_date"] - bomb["bomb_date"]
        ).dt.total_seconds() / 86400

        matched = nearby[
            (nearby["delta_days"] >= 0) &
            (nearby["delta_days"] <= LAG_DAYS)
        ]

        for _, edit in matched.iterrows():
            pairs.append({
                "bomb_idx":        idx,
                "bomb_date":       bomb["bomb_date"],
                "bomb_lon":        bomb.geometry.centroid.x,
                "bomb_lat":        bomb.geometry.centroid.y,
                "edit_osm_id":     edit.get("osm_id"),
                "edit_date":       edit["edit_date"],
                "delta_days":      round(edit["delta_days"], 1),
                "contribution_type": edit.get("contribution_type", ""),
            })

    df_pairs = pd.DataFrame(pairs)
    print(f"  Matched pairs: {len(df_pairs)}")

    # Count responses per bombing
    if not df_pairs.empty:
        counts = df_pairs.groupby("bomb_idx").size().rename("n_osm_response")
        gdf_acled = gdf_acled.join(counts)
        gdf_acled["n_osm_response"] = gdf_acled["n_osm_response"].fillna(0).astype(int)
    else:
        gdf_acled["n_osm_response"] = 0

    print(f"  Bombings with ≥1 response: {(gdf_acled['n_osm_response'] > 0).sum()}")
    return gdf_acled, df_pairs


# ── VISUALISATION ─────────────────────────────────────────────────────────────
def make_figures(gdf_acled, df_pairs, gdf_osm):
    """Produce the two key figures for the presentation."""

    # ── Fig 1: Map of Kyiv – bombings + OSM response ──────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(30.2, 30.9)
    ax.set_ylim(50.2, 50.7)

    # OSM edits (all) — light background
    gdf_osm.plot(ax=ax, color="#3498DB", markersize=2, alpha=0.2,
                 label="OSM edits (all buildings)")

    # ACLED bombings — sized by response count
    if "n_osm_response" in gdf_acled.columns:
        sizes  = np.clip(gdf_acled["n_osm_response"] * 15 + 20, 20, 300)
        colors = ["#E74C3C" if n > 0 else "#95A5A6"
                  for n in gdf_acled["n_osm_response"]]
        gdf_acled.plot(ax=ax, color=colors, markersize=sizes,
                       alpha=0.7, zorder=5)
        # Manual legend proxies
        import matplotlib.lines as mlines
        p1 = mlines.Line2D([], [], marker="o", color="w",
                           markerfacecolor="#E74C3C", markersize=10,
                           label="Bombing with OSM response (≤7 days, ≤500 m)")
        p2 = mlines.Line2D([], [], marker="o", color="w",
                           markerfacecolor="#95A5A6", markersize=7,
                           label="Bombing with no OSM response")
        p3 = mlines.Line2D([], [], marker="o", color="w",
                           markerfacecolor="#3498DB", markersize=5,
                           label="OSM building edit")
        ax.legend(handles=[p1, p2, p3], fontsize=9, loc="lower right")
    else:
        gdf_acled.plot(ax=ax, color="#E74C3C", markersize=30, alpha=0.7)

    if HAS_CTX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.Positron, zoom=12)
        except Exception as e:
            print(f"  basemap: {e}")

    ax.set_title(
        "Kyiv — Bombings (ACLED) vs OSM building edits\n"
        f"Hypothesis: edit within {RADIUS_M} m and {LAG_DAYS} days after bombing | Feb–May 2022",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(ls="--", alpha=0.25)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "fig_kyiv_map_response.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"✔ {out1}")
    plt.close(fig)

    # ── Fig 2: Time series — bombings vs OSM response count ──────────────────
    if df_pairs.empty:
        return

    daily_osm   = (df_pairs.groupby(df_pairs["bomb_date"].dt.date)
                            .size().rename("n_responses"))
    daily_bombs = (gdf_acled.groupby(
                       gdf_acled["bomb_date"].dt.date).size().rename("n_bombings"))
    ts = pd.concat([daily_bombs, daily_osm], axis=1).fillna(0)
    ts.index = pd.to_datetime(ts.index)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.bar(ts.index, ts["n_bombings"], color="#E74C3C", alpha=0.7,
            width=0.8, label="Bombing events (ACLED)")
    ax2.plot(ts.index, ts["n_responses"], color="#3498DB",
             marker="o", lw=2, ms=5, label=f"OSM edits within {LAG_DAYS}d & {RADIUS_M}m")

    ax1.set_ylabel("Bombing events / day", color="#E74C3C", fontsize=11)
    ax2.set_ylabel(f"OSM responses / day", color="#3498DB", fontsize=11)
    ax1.set_title(
        f"Temporal correlation: ACLED bombings vs OSM building edits — Kyiv, 2022\n"
        f"(OSM response = edit within {RADIUS_M} m and {LAG_DAYS} days after bombing)",
        fontsize=12, fontweight="bold")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9)
    ax1.grid(ls="--", alpha=0.25)
    plt.tight_layout()

    out2 = os.path.join(OUTPUT_DIR, "fig_kyiv_temporal_response.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"✔ {out2}")
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  OSM × ACLED — Spatio-temporal analysis, Kyiv 2022")
    print("=" * 55)

    if not os.path.exists(ACLED_FILE):
        print(f"ERROR: ACLED file not found: {ACLED_FILE}")
        print("Place your dataACLED.shp in the same folder as this script.")
        return

    if not os.path.exists(OSM_FILE):
        print(f"ERROR: OSM edits file not found: {OSM_FILE}")
        print("Run fetch_kyiv_osm_edits.py first to generate it.")
        return

    gdf_acled         = load_acled_kyiv()
    gdf_osm           = load_osm_edits()
    gdf_enriched, df_pairs = spatiotemporal_join(gdf_acled, gdf_osm)

    # Save outputs
    gdf_enriched.to_file(
        os.path.join(OUTPUT_DIR, "acled_kyiv_with_osm_response.geojson"),
        driver="GeoJSON")
    print("✔ acled_kyiv_with_osm_response.geojson")

    if not df_pairs.empty:
        df_pairs.to_csv(
            os.path.join(OUTPUT_DIR, "matched_pairs_kyiv.csv"), index=False)
        print("✔ matched_pairs_kyiv.csv")

        r = np.corrcoef(
            gdf_enriched["n_osm_response"].values,
            np.ones(len(gdf_enriched))  # placeholder
        )
        print(f"\nBombings with OSM response : "
              f"{(gdf_enriched['n_osm_response'] > 0).sum()} / {len(gdf_enriched)}")
        print(f"Total matched OSM edits    : {len(df_pairs)}")
        print(f"Mean lag (days)            : {df_pairs['delta_days'].mean():.1f}")

    make_figures(gdf_enriched, df_pairs, gdf_osm)

    print("\n" + "=" * 55)
    print(f"  Done. All outputs in ./{OUTPUT_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
