#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         OSM À L'ÉPREUVE DE LA GUERRE — Analyse principale           ║
║              Zone : Ukraine entière (4 régions)                     ║
║              Période : Fév 2022 → aujourd'hui                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Projet [anonymized] / Université Grenoble Alpes                            ║
║  Commanditaire : Raphaël Bres                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Stratégie anti-MemoryError :
  - Ukraine découpée en 4 régions traitées séquentiellement
  - Grille 0.5° (~45 km) pour les requêtes géométriques
  - Cache JSON par région dans data/cache_ohsome/
    → reprise automatique si interruption, pas de re-requête

Dépendances :
    pip install requests geopandas shapely matplotlib pandas numpy contextily
"""

import os, sys, time, logging, json
from datetime import datetime, timedelta, date

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
    print("[INFO] contextily non disponible")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

START = "2022-02-01"
END   = "2025-10-01"   # aujourd'hui dynamique
INTERVAL = "P1M"

# 4 régions Ukraine : couvrent tout le territoire
REGIONS = {
    "Ouest":  "22.0,47.5,30.0,52.5",
    "Centre": "30.0,47.5,35.0,52.5",
    "Est":    "35.0,46.0,38.5,50.0",
    "Sud":    "30.0,44.0,37.0,47.5",
}

FILTER_BUILDINGS = "building=* and type:way"
FILTER_DESTROYED = (
    "(building=ruins or ruins=yes or ruins=* "
    "or destroyed:building=* or disused:building=*) and type:way"
)

ACLED_FILE  = "dataACLED.shp"
OUTPUT_DIR  = "outputs_all_ukr"
DATA_DIR    = "data"
CACHE_DIR   = os.path.join(DATA_DIR, "cache_ohsome")

ACLED_WINDOW_DAYS = 7
GRID_RES          = 0.5    # ~45 km
UTM_CRS           = "EPSG:32637"

REGION_COLORS = {
    "Ouest": "#2980B9", "Centre": "#27AE60",
    "Est":   "#E74C3C", "Sud":    "#F39C12"
}

# =============================================================================
# UTILITAIRES
# =============================================================================

def _strip_tz(s: pd.Series) -> pd.Series:
    """Supprime le timezone d'une Series datetime — robuste à toute source."""
    try:
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def _ohsome(endpoint: str, params: dict) -> dict:
    """POST ohsome avec 3 tentatives."""
    url = f"https://api.ohsome.org/v1/{endpoint}"
    for attempt in range(3):
        try:
            r = requests.post(url, data=params, timeout=620)
            if r.status_code == 200:
                return r.json()
            log.warning(f"ohsome {r.status_code} ({attempt+1}/3): {r.text[:120]}")
        except requests.RequestException as e:
            log.warning(f"ohsome réseau ({attempt+1}/3): {e}")
        time.sleep(5)
    return {}


def _cache(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.json")


def _load(name: str):
    p = _cache(name)
    return json.load(open(p)) if os.path.exists(p) else None


def _save(name: str, data):
    json.dump(data, open(_cache(name), "w"))


# =============================================================================
# 1. LIGNE DE FRONT ISW
# =============================================================================

ISW_BASE = ("https://gist.githubusercontent.com/Viglino/"
            "675e3551fb4e79d03ac0cdb1bed2677e/raw")


def fetch_frontline(target_date: str = None) -> gpd.GeoDataFrame:
    """Snapshot ISW le plus proche de target_date (fallback 7 jours)."""
    target = target_date or END
    os.makedirs(DATA_DIR, exist_ok=True)
    for delta in range(8):
        d     = (pd.Timestamp(target) - timedelta(days=delta)).strftime("%Y-%m-%d")
        cache = os.path.join(DATA_DIR, f"isw_{d}.geojson")
        if os.path.exists(cache):
            gdf = gpd.read_file(cache)
            if not gdf.empty:
                log.info(f"ISW {target} → cache ({d})")
                return _norm_isw(gdf)
        try:
            r = requests.get(f"{ISW_BASE}/UKR-{d}.geojson", timeout=15)
            if r.status_code == 200:
                open(cache, "wb").write(r.content)
                gdf = gpd.read_file(cache)
                if not gdf.empty:
                    log.info(f"ISW {target} → téléchargé ({d})")
                    return _norm_isw(gdf)
        except requests.RequestException:
            pass
        time.sleep(0.3)
    log.warning(f"ISW {target} introuvable")
    return gpd.GeoDataFrame()


def _norm_isw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


# =============================================================================
# 2. ACLED
# =============================================================================

def load_acled() -> gpd.GeoDataFrame:
    log.info(f"Chargement ACLED : {ACLED_FILE}")
    gdf = gpd.read_file(ACLED_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    date_col = next((c for c in gdf.columns if "date" in c or "timestamp" in c), None)
    if date_col:
        gdf["date"] = pd.to_datetime(gdf[date_col], errors="coerce", utc=True).dt.tz_localize(None)

    for col in ["country", "pays", "adm0_name", "admin0"]:
        if col in gdf.columns:
            gdf = gdf[gdf[col].str.contains("Ukraine", case=False, na=False)]
            break

    if "date" in gdf.columns:
        gdf = gdf[(gdf["date"] >= pd.Timestamp(START)) &
                  (gdf["date"] <= pd.Timestamp(END))]

    log.info(f"ACLED : {len(gdf)} événements")
    return gdf.reset_index(drop=True)


# =============================================================================
# 3. OHSOME — par région avec cache
# =============================================================================

def _fetch_region(signal: str, bbox: str, region: str, extra_params: dict = None) -> list:
    """Requête ohsome pour une région, avec cache automatique."""
    key = f"{signal}_{region.lower()}"
    cached = _load(key)
    if cached is not None:
        log.info(f"  {region} [{signal}] → cache")
        return cached

    endpoints = {
        "deletions": ("contributions/count",
                      {"contributionType": "deletion", "filter": FILTER_BUILDINGS}),
        "ruins":     ("elements/count",
                      {"filter": FILTER_DESTROYED}),
        "activity":  ("contributions/count",
                      {"filter": FILTER_BUILDINGS}),
    }
    endpoint, base_params = endpoints[signal]
    params = {
        "bboxes":  bbox,
        "time":    f"{START}/{END}/{INTERVAL}",
        "timeout": "600",
        **base_params,
    }
    if extra_params:
        params.update(extra_params)

    data = _ohsome(endpoint, params)
    rows = data.get("result", [])
    _save(key, rows)
    return rows


def _build_df(rows: list, signal: str, region: str) -> pd.DataFrame:
    """Convertit les rows ohsome en DataFrame normalisé."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ts_col = "timestamp" if signal == "ruins" else "fromTimestamp"
    val_col = "n_ruines" if signal == "ruins" else (
              "deletions" if signal == "deletions" else "activity")
    df["period"] = _strip_tz(pd.to_datetime(df[ts_col], errors="coerce", utc=True))
    df[val_col]  = df["value"].fillna(0).astype(int)
    df["region"] = region
    return df[["period", val_col, "region"]]


def fetch_signal(signal: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Récupère un signal OSM pour toutes les régions.
    Retourne (DataFrame national agrégé, DataFrame par région).
    """
    log.info(f"Signal '{signal}' — 4 régions…")
    val_col = "n_ruines" if signal == "ruins" else (
              "deletions" if signal == "deletions" else "activity")
    frames = []
    for region, bbox in REGIONS.items():
        rows = _fetch_region(signal, bbox, region)
        df   = _build_df(rows, signal, region)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    nat_df = (all_df.groupby("period", as_index=False)[val_col]
                    .sum().sort_values("period").reset_index(drop=True))
    log.info(f"  {signal} → {nat_df[val_col].sum()} total, {len(nat_df)} mois")
    return nat_df, all_df


def fetch_grid_deletions() -> gpd.GeoDataFrame:
    """
    Suppressions géolocalisées — grille 0.5° sur l'Ukraine entière.
    Une requête par cellule, cache cellule par cellule.
    """
    log.info("Grille suppressions 0.5° — Ukraine entière…")
    lon_min, lat_min, lon_max, lat_max = 22.0, 44.0, 40.5, 52.5
    lons = np.arange(lon_min, lon_max, GRID_RES)
    lats = np.arange(lat_min, lat_max, GRID_RES)
    log.info(f"  {len(lons)*len(lats)} cellules")

    URL     = "https://api.ohsome.org/v1/contributions/count"
    records = []

    for i, lon in enumerate(lons):
        for lat in lats:
            key     = f"grid_{lon:.2f}_{lat:.2f}"
            cache_f = os.path.join(CACHE_DIR, f"{key}.json")

            if os.path.exists(cache_f):
                n_del = json.load(open(cache_f)).get("n", 0)
            else:
                cell = (f"{lon:.4f},{lat:.4f},"
                        f"{min(lon+GRID_RES,lon_max):.4f},"
                        f"{min(lat+GRID_RES,lat_max):.4f}")
                params = {
                    "bboxes": cell, "time": f"{START}/{END}",
                    "filter": FILTER_BUILDINGS,
                    "contributionType": "deletion", "timeout": "60"
                }
                try:
                    r = requests.post(URL, data=params, timeout=70)
                    n_del = int(sum(
                        row.get("value", 0) or 0
                        for row in (r.json().get("result", []) if r.status_code == 200 else [])
                    ))
                except Exception:
                    n_del = 0
                json.dump({"n": n_del}, open(cache_f, "w"))

            if n_del > 0:
                records.append({
                    "geometry":    Point(lon + GRID_RES/2, lat + GRID_RES/2),
                    "n_deletions": n_del
                })

        if (i + 1) % 10 == 0:
            done = (i+1) * len(lats)
            log.info(f"  {done}/{len(lons)*len(lats)} cellules — {len(records)} actives")

    if not records:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(records, crs=4326)
    log.info(f"Grille : {len(gdf)} cellules avec suppressions")
    return gdf


# =============================================================================
# 4. CROISEMENT OSM × ACLED
# =============================================================================

def correlate(df_del: pd.DataFrame, gdf_acled: gpd.GeoDataFrame) -> pd.DataFrame:
    if gdf_acled.empty or "date" not in gdf_acled.columns or df_del.empty:
        return df_del
    acled_dates = _strip_tz(gdf_acled["date"])
    periods     = _strip_tz(pd.to_datetime(df_del["period"], errors="coerce"))
    fatal_col   = next((c for c in gdf_acled.columns if "fatal" in c or "death" in c), None)
    counts, fatals = [], []
    for pe in periods:
        if pd.isna(pe):
            counts.append(0); fatals.append(0); continue
        ws   = pe - timedelta(days=ACLED_WINDOW_DAYS)
        mask = (acled_dates >= ws) & (acled_dates <= pe)
        sub  = gdf_acled[mask]
        counts.append(len(sub))
        fatals.append(pd.to_numeric(sub[fatal_col], errors="coerce").sum()
                      if fatal_col else 0)
    df = df_del.copy()
    df["n_acled_events"]     = counts
    df["n_acled_fatalities"] = fatals
    return df


# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

INV = pd.Timestamp("2022-02-24")

def _fmt(ax, interval=3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha="right")
    ax.axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7, label="Invasion 24 fév.")
    ax.grid(ls="--", alpha=0.3)


def plot_all(nat: dict, reg: dict, df_corr, gdf_acled, gdf_grid, front_gdf):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    C = {"del": "#E74C3C", "ruins": "#E67E22", "act": "#3498DB",
         "acled": "#8E44AD", "front": "#27AE60", "gray": "#95A5A6"}

    period_label = pd.Timestamp(END).strftime("%b %Y")

    # ── Fig 1 : 3 signaux nationaux ───────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(f"Signaux OSM — Ukraine entière | Fév 2022 → {period_label}",
                 fontsize=15, fontweight="bold", y=0.99)

    if not nat["activity"].empty:
        axes[0].bar(nat["activity"]["period"], nat["activity"]["activity"],
                    color=C["act"], alpha=0.8, width=25)
        axes[0].set_ylabel("Contributions / mois")
        axes[0].set_title("① Activité globale OSM (bâtiments)", fontsize=11)
        axes[0].axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7)
        axes[0].grid(axis="y", ls="--", alpha=0.3)

    if not nat["deletions"].empty:
        axes[1].bar(nat["deletions"]["period"], nat["deletions"]["deletions"],
                    color=C["del"], alpha=0.85, width=25)
        pk = nat["deletions"].loc[nat["deletions"]["deletions"].idxmax()]
        axes[1].annotate(
            f"Pic : {int(pk['deletions'])} suppressions",
            xy=(pk["period"], pk["deletions"]),
            xytext=(pk["period"], pk["deletions"] * 1.1),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        axes[1].set_ylabel("Suppressions / mois")
        axes[1].set_title("② Bâtiments supprimés (signal de destruction)", fontsize=11)
        axes[1].axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7)
        axes[1].grid(axis="y", ls="--", alpha=0.3)

    if not nat["ruins"].empty:
        axes[2].plot(nat["ruins"]["period"], nat["ruins"]["n_ruines"],
                     color=C["ruins"], marker="o", lw=2, ms=5)
        axes[2].fill_between(nat["ruins"]["period"], nat["ruins"]["n_ruines"],
                              alpha=0.15, color=C["ruins"])
        axes[2].set_ylabel("Nbre de bâtiments")
        axes[2].set_title("③ Bâtiments tagués ruins/destroyed", fontsize=11)
        axes[2].axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7)
        axes[2].grid(axis="y", ls="--", alpha=0.3)

    _fmt(axes[2])
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_signaux_ukraine.png"),
                dpi=150, bbox_inches="tight")
    log.info("✔ Fig 1"); plt.close(fig)

    # ── Fig 2 : Suppressions par région ───────────────────────────────────────
    if not reg["deletions"].empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=False)
        fig.suptitle(f"Suppressions OSM par région — Ukraine | Fév 2022 → {period_label}",
                     fontsize=14, fontweight="bold")
        for idx, (region, color) in enumerate(REGION_COLORS.items()):
            ax  = axes[idx // 2][idx % 2]
            sub = reg["deletions"][reg["deletions"]["region"] == region]
            if not sub.empty:
                ax.bar(sub["period"], sub["deletions"], color=color, alpha=0.8, width=22)
            ax.axvline(INV, color="red", lw=1.2, ls="--", alpha=0.7)
            ax.set_title(f"Région {region}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Suppressions / mois", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha="right")
            ax.grid(axis="y", ls="--", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig2_regions.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 2"); plt.close(fig)

    # ── Fig 3 : OSM vs ACLED ──────────────────────────────────────────────────
    if not df_corr.empty and "n_acled_events" in df_corr.columns:
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.bar(df_corr["period"], df_corr["deletions"],
                color=C["del"], alpha=0.7, width=20, label="Suppressions OSM")
        ax2.plot(df_corr["period"], df_corr["n_acled_events"],
                 color=C["acled"], marker="D", lw=2, ms=6,
                 label=f"Événements ACLED (J-{ACLED_WINDOW_DAYS})")
        ax1.set_ylabel("Suppressions OSM / mois", color=C["del"], fontsize=11)
        ax2.set_ylabel("Événements ACLED", color=C["acled"], fontsize=11)
        ax1.set_title(
            f"Suppressions OSM vs combats ACLED — Ukraine | Fév 2022 → {period_label}",
            fontsize=13, fontweight="bold")
        if len(df_corr) > 2:
            r = df_corr["deletions"].corr(df_corr["n_acled_events"])
            ax1.text(0.98, 0.95, f"r Pearson = {r:.2f}",
                     transform=ax1.transAxes, ha="right", va="top", fontsize=11,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                               ec=C["acled"], alpha=0.9))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
        _fmt(ax1)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig3_osm_vs_acled.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 3"); plt.close(fig)

    # ── Fig 4 : Carte Ukraine ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(22.0, 40.5); ax.set_ylim(44.0, 52.5)

    if not front_gdf.empty:
        front_gdf.plot(ax=ax, color=C["front"], alpha=0.15, zorder=2)
        front_gdf.boundary.plot(ax=ax, color=C["front"], lw=2,
                                label="Zone occupée (ISW)", zorder=3)

    if not gdf_grid.empty:
        vmax = max(gdf_grid["n_deletions"].quantile(0.97), 1)
        squares = gpd.GeoDataFrame(
            {"n_deletions": gdf_grid["n_deletions"].values},
            geometry=[sbox(p.x - GRID_RES/2, p.y - GRID_RES/2,
                           p.x + GRID_RES/2, p.y + GRID_RES/2)
                      for p in gdf_grid.geometry],
            crs=4326
        )
        squares.plot(column="n_deletions", ax=ax,
                     cmap="YlOrRd", alpha=0.75,
                     norm=Normalize(vmin=0, vmax=vmax),
                     legend=True,
                     legend_kwds={"label": "Suppressions OSM / cellule (~45 km)",
                                  "shrink": 0.5},
                     zorder=4)

    if not gdf_acled.empty:
        fatal_col = next((c for c in gdf_acled.columns
                          if "fatal" in c or "death" in c), None)
        sizes = (np.clip(
            pd.to_numeric(gdf_acled[fatal_col], errors="coerce").fillna(1) * 0.4,
            1, 15) if fatal_col else 3)
        gdf_acled.plot(ax=ax, color=C["acled"], markersize=sizes,
                       alpha=0.25, label="Événements ACLED", zorder=5)

    if HAS_CTX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.Positron, zoom=7)
        except Exception as e:
            log.warning(f"Fond de carte : {e}")

    ax.set_title(
        f"Carte des destructions OSM — Ukraine, Fév 2022 → {period_label}\n"
        "Grille suppressions  +  ACLED  +  Ligne de front ISW",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(ls="--", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_carte_ukraine.png"),
                dpi=150, bbox_inches="tight")
    log.info("✔ Fig 4"); plt.close(fig)

    # ── Fig 5 : Activité par région ────────────────────────────────────────────
    if not reg["activity"].empty:
        fig, ax = plt.subplots(figsize=(15, 6))
        for region, color in REGION_COLORS.items():
            sub = reg["activity"][reg["activity"]["region"] == region].sort_values("period")
            if not sub.empty:
                ax.plot(sub["period"], sub["activity"],
                        color=color, lw=2, marker="o", ms=4,
                        label=f"Région {region}", alpha=0.85)
        ax.set_title(f"Activité OSM par région — Ukraine | Fév 2022 → {period_label}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Contributions / mois")
        _fmt(ax)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig5_activite_regions.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 5"); plt.close(fig)

    log.info(f"Toutes les figures → '{OUTPUT_DIR}/'")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("═" * 60)
    log.info("  OSM À L'ÉPREUVE DE LA GUERRE — Ukraine entière")
    log.info(f"  Période : {START} → {END}")
    log.info(f"  Régions : {', '.join(REGIONS.keys())}")
    log.info("═" * 60)

    for d in [OUTPUT_DIR, DATA_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Ligne de front ISW ───────────────────────────────────────────────────
    front_gdf = fetch_frontline(target_date=END)
    if not front_gdf.empty:
        front_gdf.to_file(os.path.join(OUTPUT_DIR, "front_isw_latest.geojson"),
                          driver="GeoJSON")
        log.info("✔ front_isw_latest.geojson")

    # 2. ACLED ────────────────────────────────────────────────────────────────
    gdf_acled = load_acled() if os.path.exists(ACLED_FILE) else gpd.GeoDataFrame()
    if not gdf_acled.empty:
        gdf_acled.to_file(os.path.join(OUTPUT_DIR, "acled_ukraine.geojson"),
                          driver="GeoJSON")
        log.info(f"✔ acled_ukraine.geojson ({len(gdf_acled)} événements)")

    # 3. Signaux ohsome ───────────────────────────────────────────────────────
    nat, reg = {}, {}
    for signal in ["deletions", "ruins", "activity"]:
        nat[signal], reg[signal] = fetch_signal(signal)

    # Sauvegarde immédiate
    for signal in ["deletions", "ruins", "activity"]:
        val_col = "n_ruines" if signal == "ruins" else signal
        for scope, df in [("national", nat[signal]), ("regions", reg[signal])]:
            if not df.empty:
                fname = f"{signal}_{scope}.csv"
                df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
                log.info(f"✔ {fname}")

    # 4. Corrélation OSM × ACLED ──────────────────────────────────────────────
    df_corr = correlate(nat["deletions"], gdf_acled)
    if not df_corr.empty:
        df_corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_osm_acled.csv"),
                       index=False)
        log.info("✔ correlation_osm_acled.csv")

    # 5. Grille spatiale ──────────────────────────────────────────────────────
    cache_grid = os.path.join(OUTPUT_DIR, "grille_suppressions_ukraine.geojson")
    if os.path.exists(cache_grid):
        log.info(f"Grille → cache : {cache_grid}")
        gdf_grid = gpd.read_file(cache_grid)
    else:
        gdf_grid = fetch_grid_deletions()
        if not gdf_grid.empty:
            gdf_grid.to_file(cache_grid, driver="GeoJSON")
            log.info(f"✔ {cache_grid}")

    # 6. Visualisations ───────────────────────────────────────────────────────
    plot_all(nat, reg, df_corr, gdf_acled, gdf_grid, front_gdf)

    # Résumé ──────────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  RÉSUMÉ FINAL")
    log.info(f"  Période    : {START} → {END} ({len(nat['deletions'])} mois)")
    if not nat["deletions"].empty:
        log.info(f"  Suppressions totales : {nat['deletions']['deletions'].sum()}")
    if not nat["ruins"].empty:
        log.info(f"  Ruins max/mois       : {nat['ruins']['n_ruines'].max()}")
    log.info(f"  Événements ACLED     : {len(gdf_acled)}")
    log.info(f"  Fichiers → {OUTPUT_DIR}/")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
