#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║     OSM À L'ÉPREUVE DE LA GUERRE — Analyse ligne de front           ║
║     3 indicateurs : distance, zones occupée/libre, buffer 30 km     ║
║     Zone : Ukraine entière | Période : Fév 2022 → aujourd'hui       ║
╚══════════════════════════════════════════════════════════════════════╝

Source ligne de front :
  Gist GitHub (Viglino) — fichiers ISW datés depuis le 24 fév. 2022
  URL : https://gist.github.com/Viglino/675e3551fb4e79d03ac0cdb1bed2677e

3 indicateurs :
  ① Distance médiane contributions OSM ↔ ligne de front par mois
  ② Contributions zone occupée vs zone libre par mois
  ③ Contributions dans buffer 30 km autour du front par mois

Stratégie anti-MemoryError :
  - Grille 0.5° (~45 km) sur l'Ukraine entière
  - Cache cellule par cellule dans data/cache_ohsome/
  - Reprise automatique si interruption

Dépendances :
    pip install requests geopandas shapely matplotlib pandas numpy
"""

import os, sys, time, logging, json, calendar
from datetime import date, timedelta

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from shapely.geometry import Point, box as sbox
from shapely.ops import unary_union

try:
    import contextily as cx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

START     = "2022-02-01"
def _get_ohsome_end() -> str:
    try:
        r = requests.get("https://api.ohsome.org/v1/metadata", timeout=10)
        if r.status_code == 200:
            end_iso = r.json()["extractRegion"]["temporalExtent"]["toTimestamp"]
            return end_iso[:10]
    except Exception:
        pass
    return "2025-10-01"

END = _get_ohsome_end()
BUFFER_M  = 30_000      # 30 km autour du front
GRID_RES  = 0.5         # ~45 km — cohérent avec buffer front
UTM_CRS   = "EPSG:32637"
OUTPUT_DIR = "outputs"
DATA_DIR   = "data"
CACHE_DIR  = os.path.join(DATA_DIR, "cache_ohsome")

FILTER_BUILDINGS = "building=* and type:way"
ISW_BASE = ("https://gist.githubusercontent.com/Viglino/"
            "675e3551fb4e79d03ac0cdb1bed2677e/raw")

# Bbox Ukraine entière
BBOX_UKR = "22.0,44.0,40.5,52.5"


def _make_monthly_dates() -> list:
    months = []
    y, m = 2022, 2
    end_date = pd.Timestamp(END)
    while (y, m) <= (end_date.year, end_date.month):
        last = calendar.monthrange(y, m)[1]
        months.append(f"{y}-{m:02d}-{last:02d}")
        m += 1
        if m > 12:
            m = 1; y += 1
    return months


MONTHLY_DATES = _make_monthly_dates()
log.info(f"Période : {MONTHLY_DATES[0]} → {MONTHLY_DATES[-1]} ({len(MONTHLY_DATES)} mois)")

# =============================================================================
# 1. LIGNE DE FRONT HISTORIQUE — ISW
# =============================================================================

def fetch_isw_snapshot(target_date: str) -> tuple:
    """
    Télécharge et met en cache le snapshot ISW pour une date donnée.
    Essaie jusqu'à 7 jours en arrière si le fichier n'existe pas.
    Retourne (GeoDataFrame, date_réelle) ou (None, None).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for delta in range(8):
        d     = (pd.Timestamp(target_date) - timedelta(days=delta)).strftime("%Y-%m-%d")
        cache = os.path.join(DATA_DIR, f"isw_{d}.geojson")

        # Depuis le cache
        if os.path.exists(cache):
            try:
                gdf = gpd.read_file(cache)
                if not gdf.empty:
                    return _norm_isw(gdf), d
            except Exception:
                pass

        # Téléchargement
        try:
            r = requests.get(f"{ISW_BASE}/UKR-{d}.geojson", timeout=15)
            if r.status_code == 200:
                open(cache, "wb").write(r.content)
                gdf = gpd.read_file(cache)
                if not gdf.empty:
                    return _norm_isw(gdf), d
        except requests.RequestException:
            pass
        time.sleep(0.3)

    return None, None


def _norm_isw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def load_all_frontlines() -> dict:
    """
    Charge tous les snapshots ISW mensuels.
    Retourne {date_str: GeoDataFrame polygone zone occupée}.
    """
    log.info(f"Chargement des {len(MONTHLY_DATES)} snapshots ISW…")
    frontlines = {}
    for target in MONTHLY_DATES:
        gdf, actual = fetch_isw_snapshot(target)
        if gdf is None:
            log.warning(f"  {target} → introuvable")
            continue
        try:
            merged = gpd.GeoDataFrame(
                {"date": [target]},
                geometry=[unary_union(gdf.geometry.values)],
                crs=4326
            )
            frontlines[target] = merged
            log.info(f"  {target} → ok (snapshot {actual})")
        except Exception as e:
            log.warning(f"  {target} → erreur fusion : {e}")
    log.info(f"Snapshots chargés : {len(frontlines)}/{len(MONTHLY_DATES)}")
    return frontlines


# =============================================================================
# 2. CONTRIBUTIONS OSM — grille 0.5° avec cache cellule
# =============================================================================

def fetch_contributions_grid() -> gpd.GeoDataFrame:
    """
    Contributions OSM par cellule 0.5° × mois sur l'Ukraine entière.
    Cache cellule par cellule → reprise automatique.

    Retourne un GeoDataFrame :
      geometry (Point centroïde), period, n_contrib
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX_UKR.split(","))
    lons    = np.arange(lon_min, lon_max, GRID_RES)
    lats    = np.arange(lat_min, lat_max, GRID_RES)
    n_total = len(lons) * len(lats)
    log.info(f"Grille : {n_total} cellules × {len(MONTHLY_DATES)} mois")

    URL     = "https://api.ohsome.org/v1/contributions/count"
    records = []

    for i, lon in enumerate(lons):
        for lat in lats:
            key     = f"cell_{lon:.2f}_{lat:.2f}"
            cache_f = os.path.join(CACHE_DIR, f"{key}.json")

            if os.path.exists(cache_f):
                rows_cached = json.load(open(cache_f))
            else:
                cell = (f"{lon:.4f},{lat:.4f},"
                        f"{min(lon+GRID_RES,lon_max):.4f},"
                        f"{min(lat+GRID_RES,lat_max):.4f}")
                params = {
                    "bboxes":  cell,
                    "time":    f"{START}/{END}/P1M",
                    "filter":  FILTER_BUILDINGS,
                    "timeout": "60"
                }
                try:
                    r = requests.post(URL, data=params, timeout=70)
                    rows_cached = r.json().get("result", []) if r.status_code == 200 else []
                except Exception:
                    rows_cached = []
                json.dump(rows_cached, open(cache_f, "w"))

            cx = lon + GRID_RES / 2
            cy = lat + GRID_RES / 2
            for row in rows_cached:
                n = int(row.get("value", 0) or 0)
                if n > 0:
                    period = pd.to_datetime(
                        row.get("fromTimestamp"), utc=True
                    ).tz_localize(None)
                    records.append({
                        "geometry":  Point(cx, cy),
                        "period":    period,
                        "n_contrib": n
                    })

        if (i + 1) % 10 == 0:
            pct = (i + 1) / len(lons) * 100
            log.info(f"  {i+1}/{len(lons)} colonnes ({pct:.0f}%) — {len(records)} cellules-mois actives")

    if not records:
        log.warning("Aucune contribution trouvée.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(records, crs=4326)
    log.info(f"Grille contributions : {len(gdf)} cellules-mois actives")
    return gdf


# =============================================================================
# 3. INDICATEURS
# =============================================================================

def indicator1_distance(gdf: gpd.GeoDataFrame, frontlines: dict) -> pd.DataFrame:
    """
    ① Distance médiane pondérée entre contributions OSM et ligne de front.
    Pour chaque mois → médiane et moyenne en km.
    """
    log.info("Indicateur ① — distance contributions ↔ front…")
    results = []
    for month_str, front_gdf in frontlines.items():
        period  = pd.Timestamp(month_str)
        monthly = gdf[gdf["period"].dt.to_period("M") == period.to_period("M")]
        if monthly.empty:
            continue
        try:
            front_proj   = front_gdf.to_crs(UTM_CRS)
            boundary     = front_proj.geometry.boundary.unary_union
            contribs_proj = monthly.to_crs(UTM_CRS)
            dist_km      = contribs_proj.geometry.apply(
                lambda pt: boundary.distance(pt) / 1000
            )
            w = monthly["n_contrib"].values
            expanded = np.repeat(dist_km.values, w.clip(1, 50).astype(int))
            results.append({
                "period":         period,
                "median_dist_km": float(np.median(expanded)),
                "mean_dist_km":   float(np.average(dist_km.values, weights=w)),
                "n_contribs":     int(w.sum())
            })
        except Exception as e:
            log.warning(f"  {month_str} : {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        log.info(f"  Distance min : {df['median_dist_km'].min():.0f} km "
                 f"({df.loc[df['median_dist_km'].idxmin(),'period'].strftime('%b %Y')})")
        log.info(f"  Distance max : {df['median_dist_km'].max():.0f} km "
                 f"({df.loc[df['median_dist_km'].idxmax(),'period'].strftime('%b %Y')})")
    return df


def indicator2_zones(gdf: gpd.GeoDataFrame, frontlines: dict) -> pd.DataFrame:
    """
    ② Contributions dans zone occupée vs zone libre par mois.
    Jointure spatiale point-dans-polygone ISW.
    """
    log.info("Indicateur ② — zones occupée/libre…")
    results = []
    for month_str, front_gdf in frontlines.items():
        period  = pd.Timestamp(month_str)
        monthly = gdf[gdf["period"].dt.to_period("M") == period.to_period("M")].copy()
        if monthly.empty:
            continue
        try:
            joined      = gpd.sjoin(monthly, front_gdf[["geometry"]],
                                    how="left", predicate="within")
            in_occ      = joined["index_right"].notna()
            n_occ       = int(monthly.loc[in_occ.values, "n_contrib"].sum())
            n_free      = int(monthly.loc[~in_occ.values, "n_contrib"].sum())
            total       = n_occ + n_free
            results.append({
                "period":         period,
                "n_occupied":     n_occ,
                "n_free":         n_free,
                "total":          total,
                "ratio_occupied": n_occ / total if total > 0 else 0
            })
        except Exception as e:
            log.warning(f"  {month_str} : {e}")
    df = pd.DataFrame(results)
    if not df.empty:
        log.info(f"  Ratio max zone occupée : {df['ratio_occupied'].max():.1%}")
    return df


def indicator3_buffer(gdf: gpd.GeoDataFrame, frontlines: dict) -> pd.DataFrame:
    """
    ③ Contributions dans buffer 30 km autour de la ligne de front.
    """
    log.info(f"Indicateur ③ — buffer {BUFFER_M//1000} km…")
    results = []
    for month_str, front_gdf in frontlines.items():
        period  = pd.Timestamp(month_str)
        monthly = gdf[gdf["period"].dt.to_period("M") == period.to_period("M")].copy()
        if monthly.empty:
            continue
        try:
            front_proj = front_gdf.to_crs(UTM_CRS)
            buf_geom   = front_proj.geometry.boundary.buffer(BUFFER_M).unary_union
            buf_gdf    = gpd.GeoDataFrame(geometry=[buf_geom],
                                           crs=UTM_CRS).to_crs(4326)
            joined     = gpd.sjoin(monthly, buf_gdf, how="left", predicate="within")
            in_buf     = joined["index_right"].notna()
            n_buf      = int(monthly.loc[in_buf.values, "n_contrib"].sum())
            n_tot      = int(monthly["n_contrib"].sum())
            results.append({
                "period":       period,
                "n_buffer":     n_buf,
                "n_total":      n_tot,
                "ratio_buffer": n_buf / n_tot if n_tot > 0 else 0
            })
        except Exception as e:
            log.warning(f"  {month_str} : {e}")
    df = pd.DataFrame(results)
    if not df.empty:
        log.info(f"  Ratio max dans buffer : {df['ratio_buffer'].max():.1%}")
    return df


# =============================================================================
# 4. VISUALISATIONS
# =============================================================================

INV = pd.Timestamp("2022-02-24")

def _fmt(ax, interval=6):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha="right")
    ax.axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7, label="Invasion 24 fév.")
    ax.grid(ls="--", alpha=0.3)


def plot_indicators(df1: pd.DataFrame, df2: pd.DataFrame,
                    df3: pd.DataFrame, frontlines: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    C  = {"dist": "#8E44AD", "occ": "#E74C3C",
          "free": "#3498DB", "buf": "#E67E22", "front": "#27AE60"}
    period_label = date.today().strftime("%b %Y")

    # ── Fig A : Distance médiane ──────────────────────────────────────────────
    if not df1.empty:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df1["period"], df1["median_dist_km"],
                color=C["dist"], marker="o", lw=2.5, ms=7,
                label="Distance médiane pondérée (km)")
        ax.fill_between(df1["period"], df1["median_dist_km"],
                        alpha=0.15, color=C["dist"])
        # Annotations min/max
        for fn, label, offset in [("idxmin", "Min", -20), ("idxmax", "Max", +20)]:
            idx = getattr(df1["median_dist_km"], fn)()
            ax.annotate(
                f"{label} : {df1.loc[idx,'median_dist_km']:.0f} km",
                xy=(df1.loc[idx,"period"], df1.loc[idx,"median_dist_km"]),
                xytext=(df1.loc[idx,"period"],
                        df1.loc[idx,"median_dist_km"] + offset),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        ax.set_title(
            f"① Distance médiane contributions OSM ↔ ligne de front (ISW)\n"
            f"Ukraine — Fév 2022 → {period_label}",
            fontsize=13, fontweight="bold")
        ax.set_ylabel("Distance médiane (km)")
        _fmt(ax)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figA_distance_front.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig A"); plt.close(fig)

    # ── Fig B : Zones occupée / libre ────────────────────────────────────────
    if not df2.empty:
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        w   = 12
        ax1.bar(df2["period"] - pd.Timedelta(days=w),
                df2["n_free"],     width=w*2, color=C["free"],
                alpha=0.8, label="Zone libre (ukrainienne)")
        ax1.bar(df2["period"] + pd.Timedelta(days=w),
                df2["n_occupied"], width=w*2, color=C["occ"],
                alpha=0.8, label="Zone occupée (russe)")
        ax2.plot(df2["period"], df2["ratio_occupied"] * 100,
                 color="black", marker="D", lw=2, ms=5,
                 ls="--", label="% contributions zone occupée")
        ax2.set_ylabel("% contributions en zone occupée")
        ax1.set_title(
            f"② Contributions OSM : zone occupée vs zone libre\n"
            f"Ukraine — Fév 2022 → {period_label}",
            fontsize=13, fontweight="bold")
        ax1.set_ylabel("Nombre de contributions")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9)
        _fmt(ax1)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figB_zones_occupee_libre.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig B"); plt.close(fig)

    # ── Fig C : Buffer 30 km ──────────────────────────────────────────────────
    if not df3.empty:
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax2 = ax1.twinx()
        ax1.bar(df3["period"], df3["n_buffer"],
                color=C["buf"], alpha=0.8, width=22,
                label=f"Dans buffer {BUFFER_M//1000} km du front")
        ax1.bar(df3["period"], df3["n_total"] - df3["n_buffer"],
                bottom=df3["n_buffer"],
                color=C["free"], alpha=0.35, width=22,
                label="Hors buffer")
        ax2.plot(df3["period"], df3["ratio_buffer"] * 100,
                 color="black", marker="o", lw=2, ms=5,
                 ls="--", label=f"% dans buffer {BUFFER_M//1000} km")
        ax2.set_ylabel("% contributions dans le buffer")
        ax1.set_title(
            f"③ Contributions OSM dans buffer {BUFFER_M//1000} km autour du front\n"
            f"Ukraine — Fév 2022 → {period_label}",
            fontsize=13, fontweight="bold")
        ax1.set_ylabel("Contributions")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9)
        _fmt(ax1)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figC_buffer_front.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig C"); plt.close(fig)

    # ── Fig D : Grille cartes évolution front (6 snapshots) ──────────────────
    snap_dates = [
        "2022-02-28", "2022-06-30", "2022-12-31",
        "2023-06-30", "2024-06-30",
        MONTHLY_DATES[-1]
    ]
    snap_dates = [d for d in snap_dates if d in frontlines]

    if snap_dates:
        n_cols  = 3
        n_rows  = (len(snap_dates) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(18, 6 * n_rows))
        axes = np.array(axes).flatten()

        for idx, month_str in enumerate(snap_dates):
            ax = axes[idx]
            ax.set_xlim(22.0, 40.5); ax.set_ylim(44.0, 52.5)
            frontlines[month_str].plot(ax=ax, color=C["front"], alpha=0.3)
            frontlines[month_str].boundary.plot(ax=ax, color=C["front"], lw=1.5)
            ax.set_title(pd.Timestamp(month_str).strftime("%B %Y"), fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(ls="--", alpha=0.2)
            if HAS_CTX:
                try:
                    cx.add_basemap(ax, crs="EPSG:4326",
                                   source=cx.providers.CartoDB.Positron, zoom=6)
                except Exception:
                    pass

        # Masquer les sous-graphiques vides
        for idx in range(len(snap_dates), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "Évolution de la ligne de front ISW — Ukraine, 2022→aujourd'hui\n"
            "(zone verte = territoire sous contrôle russe)",
            fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figD_evolution_front.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig D"); plt.close(fig)

    log.info(f"Figures → '{OUTPUT_DIR}/'")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("═" * 60)
    log.info("  OSM vs LIGNE DE FRONT — Ukraine entière")
    log.info(f"  Période : {START} → {END} ({len(MONTHLY_DATES)} mois)")
    log.info(f"  Buffer  : {BUFFER_M//1000} km | Grille : {GRID_RES}° (~45 km)")
    log.info("═" * 60)

    for d in [OUTPUT_DIR, DATA_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Snapshots ISW ────────────────────────────────────────────────────────
    frontlines = load_all_frontlines()
    if not frontlines:
        log.error("Aucun snapshot ISW disponible — arrêt.")
        sys.exit(1)

    # 2. Contributions par cellule (avec cache) ───────────────────────────────
    cache_contribs = os.path.join(DATA_DIR, "contributions_grid_ukraine.geojson")
    if os.path.exists(cache_contribs):
        log.info(f"Contributions → cache : {cache_contribs}")
        gdf = gpd.read_file(cache_contribs)
        gdf["period"] = pd.to_datetime(gdf["period"], errors="coerce")
    else:
        gdf = fetch_contributions_grid()
        if not gdf.empty:
            gdf.to_file(cache_contribs, driver="GeoJSON")
            log.info(f"✔ {cache_contribs}")

    if gdf.empty:
        log.error("Aucune contribution disponible — arrêt.")
        sys.exit(1)

    # 3. Calcul des indicateurs ────────────────────────────────────────────────
    df1 = indicator1_distance(gdf, frontlines)
    if not df1.empty:
        df1.to_csv(os.path.join(OUTPUT_DIR, "ind1_distance_front.csv"), index=False)
        log.info("✔ ind1_distance_front.csv")

    df2 = indicator2_zones(gdf, frontlines)
    if not df2.empty:
        df2.to_csv(os.path.join(OUTPUT_DIR, "ind2_zones.csv"), index=False)
        log.info("✔ ind2_zones.csv")

    df3 = indicator3_buffer(gdf, frontlines)
    if not df3.empty:
        df3.to_csv(os.path.join(OUTPUT_DIR, "ind3_buffer.csv"), index=False)
        log.info("✔ ind3_buffer.csv")

    # 4. Visualisations ───────────────────────────────────────────────────────
    plot_indicators(df1, df2, df3, frontlines)

    # Résumé ──────────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  RÉSUMÉ")
    log.info(f"  Snapshots ISW      : {len(frontlines)}/{len(MONTHLY_DATES)} mois")
    log.info(f"  Cellules-mois OSM  : {len(gdf)}")
    if not df1.empty:
        log.info(f"  ① Distance min     : {df1['median_dist_km'].min():.0f} km")
        log.info(f"  ① Distance max     : {df1['median_dist_km'].max():.0f} km")
    if not df2.empty:
        log.info(f"  ② Ratio occ. max   : {df2['ratio_occupied'].max():.1%}")
    if not df3.empty:
        log.info(f"  ③ Ratio buffer max : {df3['ratio_buffer'].max():.1%}")
    log.info(f"  Fichiers → {OUTPUT_DIR}/")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
