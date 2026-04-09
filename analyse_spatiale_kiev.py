#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         OSM À L'ÉPREUVE DE LA GUERRE — Kiev                         ║
║              Zone : Oblast de Kiev (~80 km autour de la ville)      ║
║              Période : Fév 2022 → Déc 2022                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Projet [anonymized] / [anonymized]                            ║
║  Commanditaire : Raphaël Bres                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Contexte Kiev vs Donetsk :
  - Kiev n'a jamais été occupée (retrait russe fin mars 2022)
  - Bombardements intenses mais zone libérée rapidement
  - Signal OSM attendu : pic de contributions humanitaires,
    pas de silence cartographique lié à l'occupation
  - Permet une COMPARAISON avec Donetsk (zone occupée)

Signaux analysés :
  1. Bâtiments SUPPRIMÉS  (contributionType=deletion)
  2. Bâtiments DÉTRUITS   (tags ruins/destroyed)
  3. ACTIVITÉ GLOBALE     (toutes contributions bâtiments)

Croisements :
  - Ligne de front ISW (siège de Kiev : fév–mars 2022)
  - Événements ACLED dans la zone

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

# ── Bbox Kiev : ville + oblast (~80 km autour) ────────────────────────────────
# Couvre : Kiev, Bucha, Irpin, Borodyanka, Hostomel, Brovary
BBOX = "29.5,49.8,31.5,51.2"

START = "2022-02-01"
END   = "2022-12-31"

INTERVAL = "P1M"

FILTER_BUILDINGS = "building=* and type:way"
FILTER_DESTROYED = (
    "(building=ruins or ruins=yes or ruins=* "
    "or destroyed:building=* or disused:building=*) and type:way"
)

ACLED_FILE  = "dataACLED.shp"
OUTPUT_DIR  = "outputs_kiev"
DATA_DIR    = "data"
CACHE_DIR   = os.path.join(DATA_DIR, "cache_ohsome")

ACLED_WINDOW_DAYS = 7
GRID_RES          = 0.15   # ~12 km — zone petite, résolution fine possible
UTM_CRS           = "EPSG:32636"   # UTM 36N — adapté pour Kiev (~30°E)

ISW_BASE = ("https://gist.githubusercontent.com/Viglino/"
            "675e3551fb4e79d03ac0cdb1bed2677e/raw")

# Phases historiques spécifiques à Kiev
PHASES_KIEV = [
    ("2022-02-24", "2022-03-30", "#E74C3C", "Siège\nde Kiev"),
    ("2022-03-30", "2022-06-01", "#F39C12", "Retrait russe\n+ reconstruction"),
    ("2022-06-01", "2022-12-31", "#27AE60", "Zone libérée\n(bombardements lointains)"),
]


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


def _ohsome(endpoint: str, params: dict) -> dict:
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


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"kiev_{name}.geojson")


def _load(name: str):
    p = _cache_path(name)
    return json.load(open(p)) if os.path.exists(p) else None


def _save(name: str, data):
    json.dump(data, open(_cache_path(name), "w"))


# =============================================================================
# 1. LIGNE DE FRONT ISW
# =============================================================================

def fetch_frontline(target_date: str) -> gpd.GeoDataFrame:
    """Snapshot ISW le plus proche de target_date (fallback 7 jours)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for delta in range(8):
        d     = (pd.Timestamp(target_date) - timedelta(days=delta)).strftime("%Y-%m-%d")
        cache = os.path.join(DATA_DIR, f"isw_{d}.geojson")
        if os.path.exists(cache):
            try:
                gdf = gpd.read_file(cache)
                if not gdf.empty:
                    log.info(f"ISW {target_date} → cache ({d})")
                    return _norm_isw(gdf)
            except Exception:
                pass
        try:
            r = requests.get(f"{ISW_BASE}/UKR-{d}.geojson", timeout=15)
            if r.status_code == 200:
                open(cache, "wb").write(r.content)
                gdf = gpd.read_file(cache)
                if not gdf.empty:
                    log.info(f"ISW {target_date} → téléchargé ({d})")
                    return _norm_isw(gdf)
        except requests.RequestException:
            pass
        time.sleep(0.3)
    log.warning(f"ISW {target_date} introuvable")
    return gpd.GeoDataFrame()


def _norm_isw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def load_frontlines_kiev() -> dict:
    """
    Charge les snapshots ISW mensuels pour 2022.
    Particulièrement important pour février-mars 2022 (siège de Kiev).
    """
    # Dates clés pour Kiev — on veut une granularité fine pendant le siège
    monthly = [
        "2022-02-28", "2022-03-15", "2022-03-31",
        "2022-04-30", "2022-05-31", "2022-06-30",
        "2022-07-31", "2022-08-31", "2022-09-30",
        "2022-10-31", "2022-11-30", "2022-12-31",
    ]
    frontlines = {}
    for target in monthly:
        gdf = fetch_frontline(target)
        if gdf is None or gdf.empty:
            continue
        try:
            # Filtrer sur la zone Kiev pour alléger
            lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
            gdf_clip = gdf.cx[lon_min:lon_max, lat_min:lat_max]
            if gdf_clip.empty:
                # Pas de zone occupée près de Kiev à cette date → zone libre
                log.info(f"  {target} → aucune zone occupée près de Kiev (zone libre)")
                frontlines[target] = gpd.GeoDataFrame()
                continue
            merged = gpd.GeoDataFrame(
                {"date": [target]},
                geometry=[unary_union(gdf_clip.geometry.values)],
                crs=4326
            )
            frontlines[target] = merged
            log.info(f"  {target} → zone occupée près de Kiev détectée")
        except Exception as e:
            log.warning(f"  {target} : {e}")
            frontlines[target] = gpd.GeoDataFrame()
    return frontlines


# =============================================================================
# 2. ACLED — filtré sur Kiev
# =============================================================================

def load_acled() -> gpd.GeoDataFrame:
    if not os.path.exists(ACLED_FILE):
        log.warning(f"{ACLED_FILE} introuvable")
        return gpd.GeoDataFrame()

    log.info(f"Chargement ACLED : {ACLED_FILE}")
    gdf = gpd.read_file(ACLED_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    date_col = next((c for c in gdf.columns if "date" in c or "timestamp" in c), None)
    if date_col:
        gdf["date"] = pd.to_datetime(
            gdf[date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)

    for col in ["country", "pays", "adm0_name", "admin0"]:
        if col in gdf.columns:
            gdf = gdf[gdf[col].str.contains("Ukraine", case=False, na=False)]
            break

    # Filtre bbox Kiev
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max]

    if "date" in gdf.columns:
        gdf = gdf[(gdf["date"] >= pd.Timestamp(START)) &
                  (gdf["date"] <= pd.Timestamp(END))]

    log.info(f"ACLED Kiev : {len(gdf)} événements")
    return gdf.reset_index(drop=True)


# =============================================================================
# 3. OHSOME — requêtes pour Kiev
# =============================================================================

def fetch_deletions() -> pd.DataFrame:
    cached = _load("deletions")
    if cached:
        log.info("Suppressions Kiev → cache")
        rows = cached
    else:
        log.info("Requête ohsome — suppressions Kiev…")
        data = _ohsome("contributions/count", {
            "bboxes": BBOX, "time": f"{START}/{END}/{INTERVAL}",
            "filter": FILTER_BUILDINGS,
            "contributionType": "deletion", "timeout": "600"
        })
        rows = data.get("result", [])
        _save("deletions", rows)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"]    = _strip_tz(pd.to_datetime(df["fromTimestamp"], errors="coerce", utc=True))
    df["deletions"] = df["value"].fillna(0).astype(int)
    log.info(f"Suppressions : {df['deletions'].sum()} sur {len(df)} mois")
    return df[["period", "deletions"]]


def fetch_ruins() -> pd.DataFrame:
    cached = _load("ruins")
    if cached:
        log.info("Ruins Kiev → cache")
        rows = cached
    else:
        log.info("Requête ohsome — tags ruins Kiev…")
        data = _ohsome("elements/count", {
            "bboxes": BBOX, "time": f"{START}/{END}/{INTERVAL}",
            "filter": FILTER_DESTROYED, "timeout": "600"
        })
        rows = data.get("result", [])
        _save("ruins", rows)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"]   = _strip_tz(pd.to_datetime(df["timestamp"], errors="coerce", utc=True))
    df["n_ruines"] = df["value"].fillna(0).astype(int)
    log.info(f"Ruins max/mois : {df['n_ruines'].max()}")
    return df[["period", "n_ruines"]]


def fetch_activity() -> pd.DataFrame:
    cached = _load("activity")
    if cached:
        log.info("Activité Kiev → cache")
        rows = cached
    else:
        log.info("Requête ohsome — activité Kiev…")
        data = _ohsome("contributions/count", {
            "bboxes": BBOX, "time": f"{START}/{END}/{INTERVAL}",
            "filter": FILTER_BUILDINGS, "timeout": "600"
        })
        rows = data.get("result", [])
        _save("activity", rows)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"]   = _strip_tz(pd.to_datetime(df["fromTimestamp"], errors="coerce", utc=True))
    df["activity"] = df["value"].fillna(0).astype(int)
    return df[["period", "activity"]]


def fetch_osm_points_precise() -> gpd.GeoDataFrame:
    """
    Récupère les coordonnées PRÉCISES de chaque modification OSM via
    contributions/centroid — un point par contribution, pas un centroïde
    de cellule de grille.

    C'est l'endpoint le plus léger pour avoir des coordonnées exactes :
    ohsome renvoie un point par contribution sans la géométrie complète.

    Filtre : bâtiments uniquement, zone Kiev, fév–avr 2022
    (siège de Kiev — période demandée par le commanditaire)

    Retourne un GeoDataFrame avec colonnes :
      geometry    : Point (coordonnées précises de la contribution)
      timestamp   : date de la contribution
      osm_type    : type de contribution (creation, modification, deletion)
    """
    log.info("Requête contributions/centroid — coordonnées précises OSM Kiev…")

    # Période restreinte au siège de Kiev pour limiter le volume
    SIEGE_START = "2022-02-24"
    SIEGE_END   = "2022-04-30"

    params = {
        "bboxes":   BBOX,
        "time":     f"{SIEGE_START}/{SIEGE_END}",
        "filter":   FILTER_BUILDINGS,
        "properties": "metadata",
        "timeout":  "300"
    }

    URL = "https://api.ohsome.org/v1/contributions/centroid"
    try:
        r = requests.post(URL, data=params, timeout=320)
        if r.status_code != 200:
            log.warning(f"contributions/centroid HTTP {r.status_code}: {r.text[:200]}")
            log.info("Fallback sur contributions/centroid par type…")
            return _fetch_centroid_by_type(SIEGE_START, SIEGE_END)

        data     = r.json()
        features = data.get("features", [])
        if not features:
            log.warning("Aucune contribution reçue.")
            return gpd.GeoDataFrame()

        log.info(f"{len(features)} contributions précises reçues")
        pts, timestamps, contrib_types = [], [], []
        for f in features:
            try:
                geom = shape(f["geometry"])
                pts.append(geom)
                props = f.get("properties", {})
                ts    = props.get("@toTimestamp",
                        props.get("@snapshotTimestamp", None))
                timestamps.append(pd.to_datetime(ts, errors="coerce"))
                contrib_types.append(props.get("@contributionType", "unknown"))
            except Exception:
                continue

        gdf = gpd.GeoDataFrame({
            "timestamp":   timestamps,
            "osm_type":    contrib_types,
            "deleted_at":  timestamps   # alias pour compatibilité
        }, geometry=pts, crs=4326)

        log.info(f"Points OSM précis : {len(gdf)} contributions géolocalisées")
        return gdf

    except (requests.RequestException, MemoryError) as e:
        log.warning(f"contributions/centroid erreur : {e}")
        return _fetch_centroid_by_type(SIEGE_START, SIEGE_END)


def _fetch_centroid_by_type(start: str, end: str) -> gpd.GeoDataFrame:
    """
    Fallback : récupère les centroïdes par type de contribution
    (deletion, creation, modification) séparément pour réduire la charge.
    """
    log.info("Fallback : centroïdes par type de contribution…")
    URL    = "https://api.ohsome.org/v1/contributions/centroid"
    frames = []

    for contrib_type in ["deletion", "creation", "modification"]:
        cache_key = f"centroid_kiev_{contrib_type}_{start}_{end}"
        cache_f   = os.path.join(CACHE_DIR, f"{cache_key}.json")

        if os.path.exists(cache_f):
            features = json.load(open(cache_f))
            log.info(f"  {contrib_type} → cache ({len(features)} features)")
        else:
            params = {
                "bboxes":           BBOX,
                "time":             f"{start}/{end}",
                "filter":           FILTER_BUILDINGS,
                "contributionType": contrib_type,
                "properties":       "metadata",
                "timeout":          "300"
            }
            try:
                r = requests.post(URL, data=params, timeout=320)
                features = r.json().get("features", []) if r.status_code == 200 else []
                json.dump(features, open(cache_f, "w"))
                log.info(f"  {contrib_type} → {len(features)} features")
            except Exception as e:
                log.warning(f"  {contrib_type} erreur : {e}")
                features = []
        # ─────────────────────────────────────────────
# EXPORT GEOJSON (AJOUT SIMPLE)
# ─────────────────────────────────────────────

        
        for f in features:
            try:
                geom = shape(f["geometry"])
                props = f.get("properties", {})
                ts    = props.get("@toTimestamp",
                        props.get("@snapshotTimestamp", None))
                frames.append({
                    "geometry":   geom,
                    "timestamp":  pd.to_datetime(ts, errors="coerce"),
                    "osm_type":   contrib_type,
                    "deleted_at": pd.to_datetime(ts, errors="coerce")
                })
            except Exception:
                continue

    if not frames:
        log.warning("Aucune contribution récupérée en fallback.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(frames, crs=4326)
    log.info(f"Fallback : {len(gdf)} contributions géolocalisées au total")
    return gdf


def fetch_deletions_geom() -> gpd.GeoDataFrame:
    """
    Pour la carte de chaleur : suppressions géolocalisées via grille 0.15°.
    Utilisé uniquement pour fig3 (carte). La corrélation précise
    utilise fetch_osm_points_precise().
    """
    log.info("Grille suppressions 0.15° — Kiev (pour la carte)…")
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
    lons = np.arange(lon_min, lon_max, GRID_RES)
    lats = np.arange(lat_min, lat_max, GRID_RES)

    URL     = "https://api.ohsome.org/v1/contributions/count"
    records = []
    for lon in lons:
        for lat in lats:
            key     = f"grid_kv_{lon:.2f}_{lat:.2f}"
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
                        for row in (r.json().get("result", [])
                                    if r.status_code == 200 else [])
                    ))
                except Exception:
                    n_del = 0
                json.dump({"n": n_del}, open(cache_f, "w"))
            if n_del > 0:
                records.append({
                    "geometry":    Point(lon + GRID_RES/2, lat + GRID_RES/2),
                    "n_deletions": n_del
                })

    if not records:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(records, crs=4326)
    log.info(f"Grille carte : {len(gdf)} cellules avec suppressions")
    return gdf


# =============================================================================
# 4. CROISEMENT OSM × ACLED — corrélation SPATIO-TEMPORELLE
# =============================================================================

# Rayon spatial autour de chaque frappe ACLED (mètres)
SPATIAL_BUFFER_M = 1000   # 1 km — ajustable

def correlate_spatiotemporal(
    gdf_osm_pts: gpd.GeoDataFrame,
    gdf_acled:   gpd.GeoDataFrame,
    window_days: int = ACLED_WINDOW_DAYS,
    buffer_m:    int = SPATIAL_BUFFER_M
) -> gpd.GeoDataFrame:
    """
    Corrélation SPATIO-TEMPORELLE bombardements ACLED × modifications OSM.

    Pour chaque événement ACLED (frappe, bataille) :
      1. Crée un buffer spatial de buffer_m mètres autour du point
      2. Cherche les modifications OSM dans ce buffer
         ET dans la fenêtre [date_ACLED, date_ACLED + window_days]
      3. Considère la paire comme "corrélée" si au moins 1 modif OSM trouvée

    Retourne un GeoDataFrame de paires corrélées avec colonnes :
      geometry       : point de la frappe ACLED
      date_acled     : date de la frappe
      event_type     : type d'événement ACLED
      n_osm_nearby   : nb de modifications OSM dans le buffer/fenêtre
      delay_days_min : délai minimal (jours) entre frappe et première modif OSM
      correlated     : True si au moins 1 modif OSM trouvée

    Sauvegarde aussi un CSV de toutes les paires pour le rapport.
    """
    if gdf_acled.empty or gdf_osm_pts.empty:
        log.warning("Données manquantes pour la corrélation spatio-temporelle.")
        return gpd.GeoDataFrame()

    log.info(f"Corrélation spatio-temporelle : {len(gdf_acled)} frappes × "
             f"{len(gdf_osm_pts)} points OSM")
    log.info(f"  Paramètres : buffer={buffer_m}m, fenêtre={window_days}j")

    # Normalisation dates
    acled_dates = _strip_tz(gdf_acled["date"])
    osm_dates   = _strip_tz(gdf_osm_pts["deleted_at"]) \
                  if "deleted_at" in gdf_osm_pts.columns \
                  else pd.Series([pd.NaT] * len(gdf_osm_pts))

    # Projection métrique pour le buffer
    gdf_acled_proj   = gdf_acled.to_crs(UTM_CRS)
    gdf_osm_proj     = gdf_osm_pts.to_crs(UTM_CRS)

    event_type_col = next(
        (c for c in gdf_acled.columns if "event" in c and "type" in c), None
    )
    fatal_col = next(
        (c for c in gdf_acled.columns if "fatal" in c or "death" in c), None
    )

    results = []
    n_correlated = 0

    for idx in range(len(gdf_acled)):
        frappe_geom  = gdf_acled_proj.geometry.iloc[idx]
        frappe_date  = acled_dates.iloc[idx]
        frappe_proj  = gdf_acled_proj.iloc[idx]

        if pd.isna(frappe_date):
            continue

        # 1. Buffer spatial autour de la frappe
        buf = frappe_geom.buffer(buffer_m)

        # 2. Filtrer OSM : dans le buffer spatial
        in_buf = gdf_osm_proj.geometry.within(buf)

        # 3. Filtrer OSM : dans la fenêtre temporelle [date, date + 7j]
        window_end   = frappe_date + timedelta(days=window_days)
        if osm_dates.notna().any():
            in_window = (osm_dates >= frappe_date) & (osm_dates <= window_end)
            mask      = in_buf & in_window.values
        else:
            # Pas de date OSM disponible → filtre spatial seulement
            mask = in_buf

        osm_nearby = gdf_osm_pts[mask]
        n_nearby   = len(osm_nearby)
        correlated = n_nearby > 0
        if correlated:
            n_correlated += 1

        # Délai minimal entre frappe et première modif OSM
        delay_min = None
        if correlated and osm_dates.notna().any():
            dates_nearby = osm_dates[mask]
            valid = dates_nearby.dropna()
            if not valid.empty:
                delay_min = float((valid - frappe_date).dt.days.min())

        results.append({
            "geometry":       gdf_acled.geometry.iloc[idx],
            "date_acled":     frappe_date,
            "event_type":     gdf_acled[event_type_col].iloc[idx] if event_type_col else None,
            "fatalities":     pd.to_numeric(
                                  gdf_acled[fatal_col].iloc[idx], errors="coerce"
                              ) if fatal_col else 0,
            "n_osm_nearby":   n_nearby,
            "delay_days_min": delay_min,
            "correlated":     correlated,
        })

    log.info(f"  Frappes corrélées : {n_correlated}/{len(gdf_acled)} "
             f"({n_correlated/len(gdf_acled)*100:.1f}%)")

    if not results:
        return gpd.GeoDataFrame()

    gdf_result = gpd.GeoDataFrame(results, crs=4326)

    # Sauvegarde CSV des paires pour le rapport
    csv_path = os.path.join(OUTPUT_DIR, "paires_acled_osm_spatio_temporel.csv")
    gdf_result.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    log.info(f"✔ Paires sauvegardées : {csv_path}")

    return gdf_result


def correlate(df_del: pd.DataFrame, gdf_acled: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Corrélation temporelle agrégée par mois (pour les graphiques de tendance).
    Complète correlate_spatiotemporal() qui travaille frappe par frappe.
    """
    if gdf_acled.empty or "date" not in gdf_acled.columns or df_del.empty:
        return df_del
    acled_dates = _strip_tz(gdf_acled["date"])
    periods     = _strip_tz(pd.to_datetime(df_del["period"], errors="coerce"))
    fatal_col   = next((c for c in gdf_acled.columns
                        if "fatal" in c or "death" in c), None)
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
RETRAIT = pd.Timestamp("2022-03-30")   # retrait russe de la région de Kiev

C = {
    "del":   "#E74C3C",
    "ruins": "#E67E22",
    "act":   "#3498DB",
    "acled": "#8E44AD",
    "front": "#27AE60",
    "gray":  "#95A5A6",
}


def _fmt(ax, interval=1):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha="right")
    ax.axvline(INV, color="red", lw=1.8, ls="--", alpha=0.8,
               label="Invasion 24 fév.")
    ax.axvline(RETRAIT, color="green", lw=1.5, ls="--", alpha=0.8,
               label="Retrait russe ~30 mars")
    ax.grid(ls="--", alpha=0.3)


def _add_phases(ax):
    """Ajoute les zones de phases de guerre spécifiques à Kiev."""
    ylim = ax.get_ylim()
    for p_start, p_end, color, label in PHASES_KIEV:
        ax.axvspan(pd.Timestamp(p_start), pd.Timestamp(p_end),
                   alpha=0.07, color=color)
        mid = pd.Timestamp(p_start) + (
            pd.Timestamp(p_end) - pd.Timestamp(p_start)) / 2
        ax.text(mid, ylim[1] * 0.92, label,
                ha="center", va="top", fontsize=8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.75))


def plot_all(df_del, df_ruins, df_act, df_corr,
             gdf_acled, gdf_grid, frontlines):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Fig 1 : 3 signaux OSM Kiev ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        "Signaux OSM — Kiev & Oblast | Fév–Déc 2022\n"
        "(siège fév–mars 2022, retrait russe fin mars, zone libérée ensuite)",
        fontsize=14, fontweight="bold", y=0.99
    )

    if not df_act.empty:
        axes[0].bar(df_act["period"], df_act["activity"],
                    color=C["act"], alpha=0.8, width=22)
        axes[0].set_ylabel("Contributions / mois")
        axes[0].set_title("① Activité globale OSM (bâtiments)", fontsize=11)
        axes[0].axvline(INV, color="red", lw=1.8, ls="--", alpha=0.8)
        axes[0].axvline(RETRAIT, color="green", lw=1.5, ls="--", alpha=0.8)
        axes[0].grid(axis="y", ls="--", alpha=0.3)
        _add_phases(axes[0])

    if not df_del.empty:
        axes[1].bar(df_del["period"], df_del["deletions"],
                    color=C["del"], alpha=0.85, width=22)
        if len(df_del) > 0:
            pk = df_del.loc[df_del["deletions"].idxmax()]
            axes[1].annotate(
                f"Pic : {int(pk['deletions'])} suppressions",
                xy=(pk["period"], pk["deletions"]),
                xytext=(pk["period"], pk["deletions"] * 1.12),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        axes[1].set_ylabel("Suppressions / mois")
        axes[1].set_title("② Bâtiments supprimés (signal de destruction physique)", fontsize=11)
        axes[1].axvline(INV, color="red", lw=1.8, ls="--", alpha=0.8)
        axes[1].axvline(RETRAIT, color="green", lw=1.5, ls="--", alpha=0.8)
        axes[1].grid(axis="y", ls="--", alpha=0.3)
        _add_phases(axes[1])

    if not df_ruins.empty:
        axes[2].plot(df_ruins["period"], df_ruins["n_ruines"],
                     color=C["ruins"], marker="o", lw=2.5, ms=7)
        axes[2].fill_between(df_ruins["period"], df_ruins["n_ruines"],
                              alpha=0.18, color=C["ruins"])
        axes[2].set_ylabel("Nbre de bâtiments")
        axes[2].set_title("③ Bâtiments tagués ruins/destroyed", fontsize=11)
        axes[2].axvline(INV, color="red", lw=1.8, ls="--", alpha=0.8)
        axes[2].axvline(RETRAIT, color="green", lw=1.5, ls="--", alpha=0.8)
        axes[2].grid(axis="y", ls="--", alpha=0.3)
        _add_phases(axes[2])

    _fmt(axes[2])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_signaux_kiev.png"),
                dpi=150, bbox_inches="tight")
    log.info("✔ Fig 1 — signaux OSM Kiev"); plt.close(fig)

    # ── Fig 2 : OSM vs ACLED Kiev ─────────────────────────────────────────────
    if not df_corr.empty and "n_acled_events" in df_corr.columns:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()
        ax1.bar(df_corr["period"], df_corr["deletions"],
                color=C["del"], alpha=0.7, width=18, label="Suppressions OSM")
        ax2.plot(df_corr["period"], df_corr["n_acled_events"],
                 color=C["acled"], marker="D", lw=2.2, ms=8,
                 label=f"Événements ACLED (J-{ACLED_WINDOW_DAYS})")
        ax2.fill_between(df_corr["period"], df_corr["n_acled_events"],
                         alpha=0.1, color=C["acled"])
        ax1.set_ylabel("Suppressions OSM / mois", color=C["del"], fontsize=11)
        ax2.set_ylabel("Événements ACLED", color=C["acled"], fontsize=11)
        ax1.set_title(
            "Suppressions OSM vs combats ACLED — Kiev, Fév–Déc 2022",
            fontsize=13, fontweight="bold"
        )
        if len(df_corr) > 2:
            r = df_corr["deletions"].corr(df_corr["n_acled_events"])
            ax1.text(0.98, 0.95, f"r Pearson = {r:.2f}",
                     transform=ax1.transAxes, ha="right", va="top",
                     fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                               ec=C["acled"], alpha=0.9))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
        _fmt(ax1)
        _add_phases(ax1)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig2_osm_vs_acled_kiev.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 2 — OSM vs ACLED Kiev"); plt.close(fig)

    # ── Fig 3 : Carte suppressions Kiev ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)

    # Ligne de front début mars (siège)
    front_siege = frontlines.get("2022-03-15", frontlines.get("2022-03-31"))
    if front_siege is not None and not front_siege.empty:
        front_siege.plot(ax=ax, color=C["front"], alpha=0.15, zorder=2)
        front_siege.boundary.plot(ax=ax, color=C["front"], lw=2,
                                   label="Zone occupée mi-mars 2022 (ISW)", zorder=3)

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
                     legend_kwds={"label": "Suppressions OSM / cellule (~12 km)",
                                  "shrink": 0.55},
                     zorder=4)

    if not gdf_acled.empty:
        fatal_col = next((c for c in gdf_acled.columns
                          if "fatal" in c or "death" in c), None)
        sizes = (np.clip(
            pd.to_numeric(gdf_acled[fatal_col], errors="coerce").fillna(1) * 1.2,
            2, 30) if fatal_col else 5)
        gdf_acled.plot(ax=ax, color=C["acled"], markersize=sizes,
                       alpha=0.5, label="Événements ACLED", zorder=5)

    if HAS_CTX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.Positron, zoom=10)
        except Exception as e:
            log.warning(f"Fond de carte : {e}")

    ax.set_title(
        "Carte des destructions OSM — Kiev & Oblast, Fév–Déc 2022\n"
        "Grille suppressions  +  ACLED  +  Front ISW (mi-mars 2022)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(ls="--", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_carte_kiev.png"),
                dpi=150, bbox_inches="tight")
    log.info("✔ Fig 3 — carte Kiev"); plt.close(fig)

    # ── Fig 4 : Types d'événements ACLED Kiev ─────────────────────────────────
    type_col = next((c for c in gdf_acled.columns
                     if "event" in c and "type" in c), None) if not gdf_acled.empty else None
    if type_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Analyse ACLED — Kiev & Oblast, Fév–Déc 2022",
                     fontsize=13, fontweight="bold")
        counts = gdf_acled[type_col].value_counts().head(8)
        counts.plot(kind="barh", ax=axes[0],
                    color=plt.cm.Set2(np.linspace(0, 1, len(counts))),
                    edgecolor="white")
        axes[0].set_title("Types d'événements"); axes[0].set_xlabel("Nombre")
        axes[0].invert_yaxis(); axes[0].grid(axis="x", ls="--", alpha=0.4)

        if "date" in gdf_acled.columns:
            top4 = counts.head(4).index.tolist()
            for typ, col in zip(top4, plt.cm.Set1(np.linspace(0, 0.8, 4))):
                sub = gdf_acled[gdf_acled[type_col] == typ]
                mo  = (sub.groupby(sub["date"].dt.to_period("M"))
                          .size().reset_index(name="n"))
                mo["date"] = mo["date"].dt.to_timestamp()
                axes[1].plot(mo["date"], mo["n"], marker="o", lw=1.8,
                             ms=5, label=typ[:25], color=col, alpha=0.85)
            axes[1].set_title("Évolution mensuelle (top 4)")
            axes[1].set_ylabel("Nb d'événements")
            axes[1].axvline(INV, color="red", lw=1.5, ls="--", alpha=0.7)
            axes[1].axvline(RETRAIT, color="green", lw=1.5, ls="--", alpha=0.7)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=40, ha="right")
            axes[1].legend(fontsize=8); axes[1].grid(ls="--", alpha=0.35)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig4_acled_types_kiev.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 4 — types ACLED Kiev"); plt.close(fig)

    # ── Fig 5 : Synthèse triple — OSM + ACLED + phases Kiev ──────────────────
    if not df_del.empty and not df_act.empty:
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 65))

        ax1.bar(df_act["period"], df_act["activity"],
                color=C["gray"], alpha=0.4, width=22, label="Activité OSM totale")
        ax1.bar(df_del["period"], df_del["deletions"],
                color=C["del"], alpha=0.85, width=15, label="Suppressions OSM")

        if not df_ruins.empty:
            ax2.plot(df_ruins["period"], df_ruins["n_ruines"],
                     color=C["ruins"], lw=2, ms=6, marker="s",
                     label="Tags ruins/destroyed")

        if not df_corr.empty and "n_acled_events" in df_corr.columns:
            mo_acled = df_corr[["period", "n_acled_events"]].copy()
            ax3.plot(mo_acled["period"], mo_acled["n_acled_events"],
                     color=C["acled"], lw=2, ms=7, marker="D",
                     ls="--", label="Événements ACLED")

        ax1.axvline(INV, color="red", lw=2, alpha=0.8, label="Invasion 24 fév.")
        ax1.axvline(RETRAIT, color="green", lw=1.8, alpha=0.8,
                    label="Retrait russe ~30 mars")

        _add_phases(ax1)

        ax1.set_ylabel("Contributions / Suppressions OSM", fontsize=10)
        ax2.set_ylabel("Bâtiments tags ruins", color=C["ruins"], fontsize=10)
        ax3.set_ylabel("Événements ACLED / mois", color=C["acled"], fontsize=10)
        ax1.set_title(
            "Synthèse : activité OSM, destructions et combats ACLED — Kiev\n"
            "Fév–Déc 2022  (siège → retrait → zone libérée)",
            fontsize=13, fontweight="bold"
        )
        _fmt(ax1)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2 + h3, l1 + l2 + l3,
                   fontsize=9, loc="upper right", framealpha=0.9, ncol=2)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig5_synthese_kiev.png"),
                    dpi=150, bbox_inches="tight")
        log.info("✔ Fig 5 — synthèse Kiev"); plt.close(fig)

    log.info(f"Toutes les figures → '{OUTPUT_DIR}/'")


def plot_spatiotemporal(gdf_pairs: gpd.GeoDataFrame, gdf_acled: gpd.GeoDataFrame):
    """
    Fig 6 — Carte et statistiques de la corrélation spatio-temporelle.
    Montre quelles frappes ACLED ont été suivies d'une modification OSM
    dans les 7 jours et dans un rayon de 1 km.
    """
    if gdf_pairs.empty:
        log.warning("Pas de paires spatio-temporelles à visualiser.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Gauche : carte des frappes corrélées vs non corrélées ────────────────
    ax = axes[0]
    lon_min, lat_min, lon_max, lat_max = map(float, BBOX.split(","))
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)

    non_corr = gdf_pairs[~gdf_pairs["correlated"]]
    corr     = gdf_pairs[gdf_pairs["correlated"]]

    if not non_corr.empty:
        non_corr.plot(ax=ax, color="#95A5A6", markersize=6,
                      alpha=0.5, label="Frappe sans réponse OSM")
    if not corr.empty:
        sizes = np.clip(corr["n_osm_nearby"].values * 8, 10, 80)
        corr.plot(ax=ax, color="#E74C3C", markersize=sizes,
                  alpha=0.8, label="Frappe avec réponse OSM (7j, 1km)", zorder=5)

    if HAS_CTX:
        try:
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.Positron, zoom=10)
        except Exception:
            pass

    total      = len(gdf_pairs)
    n_corr     = corr.shape[0]
    pct        = n_corr / total * 100 if total > 0 else 0
    ax.set_title(
        f"Corrélation spatio-temporelle ACLED × OSM — Kiev\n"
        f"Fév–Avr 2022 | {n_corr}/{total} frappes ({pct:.0f}%) "
        f"suivies d'une modif OSM\n(fenêtre : {ACLED_WINDOW_DAYS}j, rayon : {SPATIAL_BUFFER_M}m)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(ls="--", alpha=0.25)

    # ── Droite : distribution des délais ─────────────────────────────────────
    ax2 = axes[1]
    delays = corr["delay_days_min"].dropna()

    if not delays.empty:
        ax2.hist(delays, bins=range(0, ACLED_WINDOW_DAYS + 2),
                 color="#E74C3C", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax2.axvline(delays.mean(), color="black", lw=2, ls="--",
                    label=f"Délai moyen : {delays.mean():.1f}j")
        ax2.axvline(delays.median(), color="#8E44AD", lw=2, ls="--",
                    label=f"Délai médian : {delays.median():.1f}j")
        ax2.set_xlabel("Délai entre frappe ACLED et modification OSM (jours)")
        ax2.set_ylabel("Nombre de paires")
        ax2.set_title(
            f"Distribution des délais\n"
            f"(sur {len(delays)} paires corrélées)",
            fontsize=11, fontweight="bold"
        )
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, ACLED_WINDOW_DAYS + 0.5)
        ax2.set_xticks(range(0, ACLED_WINDOW_DAYS + 1))
        ax2.grid(axis="y", ls="--", alpha=0.35)
    else:
        ax2.text(0.5, 0.5, "Pas de données de délai disponibles\n"
                 "(dates OSM non renseignées)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig6_correlation_spatiotemporelle.png"),
                dpi=150, bbox_inches="tight")
    log.info("✔ Fig 6 — corrélation spatio-temporelle"); plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("═" * 60)
    log.info("  OSM À L'ÉPREUVE DE LA GUERRE — KIEV")
    log.info(f"  Bbox    : {BBOX}")
    log.info(f"  Période : {START} → {END}")
    log.info("═" * 60)

    for d in [OUTPUT_DIR, DATA_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Snapshots ISW Kiev ───────────────────────────────────────────────────
    log.info("Chargement snapshots ISW pour Kiev…")
    frontlines = load_frontlines_kiev()

    # 2. ACLED filtré Kiev ────────────────────────────────────────────────────
    gdf_acled = load_acled()
    if not gdf_acled.empty:
        gdf_acled.to_file(os.path.join(OUTPUT_DIR, "acled_kiev.geojson"),
                          driver="GeoJSON")
        log.info(f"✔ acled_kiev.geojson ({len(gdf_acled)} événements)")

    # 3. Requêtes ohsome Kiev ─────────────────────────────────────────────────
    df_del   = fetch_deletions()
    df_ruins = fetch_ruins()
    df_act   = fetch_activity()

    for df, fname in [
        (df_del,   "deletions_kiev.csv"),
        (df_ruins, "ruins_kiev.csv"),
        (df_act,   "activite_kiev.csv"),
    ]:
        if not df.empty:
            df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
            log.info(f"✔ {fname}")

    # 4a. Corrélation temporelle agrégée (graphiques de tendance)
    df_corr = correlate(df_del, gdf_acled)
    if not df_corr.empty:
        df_corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_kiev.csv"), index=False)
        log.info("✔ correlation_kiev.csv")

    # 4b. Corrélation SPATIO-TEMPORELLE point à point (demande commanditaire)
    # Coordonnées précises frappe ACLED × coordonnées précises modif OSM
    log.info("Corrélation point à point (frappe × modif OSM dans 1km/7j)…")
    osm_source = gdf_osm_pts if not gdf_osm_pts.empty else gdf_grid
    if not osm_source.empty and not gdf_acled.empty:
        acled_siege = gdf_acled.copy()
        if "date" in acled_siege.columns:
            acled_siege = acled_siege[
                (acled_siege["date"] >= pd.Timestamp("2022-02-24")) &
                (acled_siege["date"] <= pd.Timestamp("2022-04-30"))
            ]
        log.info(f"  ACLED siège de Kiev : {len(acled_siege)} événements")
        gdf_pairs = correlate_spatiotemporal(osm_source, acled_siege)
        if not gdf_pairs.empty:
            gdf_pairs.to_file(
                os.path.join(OUTPUT_DIR, "paires_spatiotemporelles.geojson"),
                driver="GeoJSON"
            )
            log.info("✔ paires_spatiotemporelles.geojson (visualisable dans QGIS)")
            plot_spatiotemporal(gdf_pairs, acled_siege)
    else:
        log.warning("Données insuffisantes pour la corrélation spatio-temporelle.")

    # 5. Points OSM PRÉCIS pour la corrélation point à point ─────────────────
    gdf_osm_pts = gpd.GeoDataFrame()   # initialisé vide par défaut
    cache_pts = os.path.join(OUTPUT_DIR, "osm_points_precis_kiev.geojson")
    if os.path.exists(cache_pts):
        log.info(f"Points précis → cache : {cache_pts}")
        gdf_osm_pts = gpd.read_file(cache_pts)
        gdf_osm_pts["timestamp"] = pd.to_datetime(
            gdf_osm_pts.get("timestamp"), errors="coerce"
        )
        gdf_osm_pts["deleted_at"] = gdf_osm_pts["timestamp"]
    else:
        gdf_osm_pts = fetch_osm_points_precise()
        if not gdf_osm_pts.empty:
            gdf_osm_pts.to_file(cache_pts, driver="GeoJSON")
            log.info(f"✔ {cache_pts} ({len(gdf_osm_pts)} points)")

    # 5b. Grille pour la carte de chaleur (fig3) ──────────────────────────────
    gdf_grid = gpd.GeoDataFrame()   # initialisé vide par défaut
    cache_grid = os.path.join(OUTPUT_DIR, "grille_suppressions_kiev.geojson")
    if os.path.exists(cache_grid):
        log.info(f"Grille → cache : {cache_grid}")
        gdf_grid = gpd.read_file(cache_grid)
    else:
        gdf_grid = fetch_deletions_geom()
        if not gdf_grid.empty:
            gdf_grid.to_file(cache_grid, driver="GeoJSON")
            log.info(f"✔ {cache_grid}")

    # 6. Visualisations ───────────────────────────────────────────────────────
    plot_all(df_del, df_ruins, df_act, df_corr,
             gdf_acled, gdf_grid, frontlines)
    OUTPUT = "osm_contributions_kiev.geojson"

    if 'gdf' in globals() and not gdf.empty:
        try:
            gdf_export = gdf.copy()
            if "date" in gdf_export.columns:
                gdf_export["date"] = gdf_export["date"].astype(str)

                gdf_export.to_file(OUTPUT, driver="GeoJSON")

                print(f"\n✅ GeoJSON généré : {OUTPUT}")
                print(f"Nombre de points : {len(gdf_export)}")

        except Exception as e:
            print(f"❌ Erreur export GeoJSON : {e}")

        else:
                print("❌ gdf vide → pas de fichier généré")
    # Résumé ──────────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  RÉSUMÉ FINAL — Kiev")
    log.info(f"  Suppressions totales : {df_del['deletions'].sum() if not df_del.empty else 'N/A'}")
    log.info(f"  Ruins max/mois       : {df_ruins['n_ruines'].max() if not df_ruins.empty else 'N/A'}")
    log.info(f"  Événements ACLED     : {len(gdf_acled)}")
    log.info(f"  Fichiers → {OUTPUT_DIR}/")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
    

