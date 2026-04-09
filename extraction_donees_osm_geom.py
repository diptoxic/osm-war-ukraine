#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OSM À L'ÉPREUVE DE LA GUERRE — Script de collecte généralisé             ║
║   Contributions OSM × Bombardements ACLED                                  ║
║   Durée complète de la guerre : fév. 2022 → présent                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Projet [anonymized] / Université Grenoble Alpes                                   ║
║  Commanditaire : Raphaël Bres                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  ZONES CONFIGURABLES (STUDY_AREAS) :                                       ║
║    kyiv      → Kiev ville + proche banlieue                                ║
║    kharkiv   → Kharkiv (zone occupée / ligne de front Est)                 ║
║    donetsk   → Donbass / Oblast Donetsk                                    ║
║    kherson   → Kherson (occupation puis libération nov. 2022)              ║
║    mariupol  → Mariupol (siège mars–mai 2022)                              ║
║    ukraine   → Ukraine entière (agrégation nationale)                      ║
║                                                                             ║
║  OUTPUTS (par zone, dans outputs/<zone>/) :                                ║
║    contributions_<zone>.geojson   ← 1 point / contribution OSM réelle     ║
║    deletions_<zone>.geojson       ← suppressions seules                    ║
║    ruins_<zone>.geojson           ← bâtiments ruins/destroyed              ║
║    acled_bombings_<zone>.geojson  ← bombardements ACLED filtrés            ║
║    match_osm_acled_<zone>.geojson ← contributions enrichies (corr. ACLED) ║
║    series_mensuelle_<zone>.csv    ← agrégation mensuelle pour graphiques   ║
║                                                                             ║
║  FILTRE ACLED BOMBARDEMENTS (corrigé) :                                    ║
║    event_type    = "Explosions/Remote violence"                            ║
║    sub_event_type ∈ air/drone strike, shelling/artillery/missile attack,   ║
║                     remote explosive/landmine/IED, missile attack,         ║
║                     guided missile strike                                  ║
║                                                                             ║
║  DÉPENDANCES :                                                              ║
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
# CONFIGURATION GLOBALE
# =============================================================================

# ── Période de la guerre ──────────────────────────────────────────────────────
WAR_START = "2022-02-24"
WAR_END   = "2024-12-31"   # adapter si nécessaire

# ── Zones d'étude ─────────────────────────────────────────────────────────────
# Chaque zone : (bbox "lon_min,lat_min,lon_max,lat_max", nom lisible)
STUDY_AREAS = {
    "kyiv":     ("30.20,50.20,30.90,50.60", "Kiev (ville)"),
    "kharkiv":  ("36.00,49.80,36.60,50.20", "Kharkiv"),
    "donetsk":  ("37.50,47.80,38.20,48.20", "Donetsk"),
    "kherson":  ("32.40,46.50,33.00,46.80", "Kherson"),
    "mariupol": ("37.40,47.00,37.80,47.20", "Marioupol"),
    "ukraine":  ("22.00,44.00,40.50,52.50", "Ukraine entière"),
}

# Zone(s) à traiter — modifier cette liste pour cibler une zone
# Ex : ZONES_TO_PROCESS = ["kyiv", "kharkiv"]
ZONES_TO_PROCESS = ["donetsk"]

# ── Fichier ACLED ──────────────────────────────────────────────────────────────
ACLED_FILE = "dataACLED.shp"   # placer à la racine du projet

# ── Répertoires ───────────────────────────────────────────────────────────────
BASE_OUTPUT = "outputs"
BASE_CACHE  = "data/cache"

# ── Paramètres ohsome ─────────────────────────────────────────────────────────
FILTER_BUILDINGS = "building=* and type:way"
FILTER_RUINS     = (
    "(building=ruins or ruins=yes or ruins=* "
    "or destroyed:building=* or disused:building=*) and type:way"
)
OHSOME_URL = "https://api.ohsome.org/v1"

# ── Paramètres corrélation spatio-temporelle ──────────────────────────────────
SPATIAL_BUFFER_M = 1000   # rayon en mètres
WINDOW_DAYS      = 7      # fenêtre temporelle post-frappe
UTM_CRS          = "EPSG:32637"   # UTM 37N — centré sur l'Ukraine

# ── Filtre ACLED bombardements ────────────────────────────────────────────────
# event_type doit contenir "Explosion"
ACLED_EVENT_TYPE_FILTER = "Explosion"

# sub_event_type doit correspondre à l'un de ces types
ACLED_BOMBING_SUBTYPES = [
    "air/drone strike",
    "shelling/artillery/missile attack",
    "remote explosive/landmine/ied",
    "missile attack",
    "guided missile strike",
]

# ── Découpage mensuel pour ohsome (évite les timeouts) ────────────────────────
def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Génère les couples (début_mois, fin_mois) entre start et end."""
    ranges = []
    cur = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        next_m = (cur + pd.offsets.MonthEnd(1)).replace(hour=0,
                                                         minute=0,
                                                         second=0,
                                                         microsecond=0)
        # Le dernier morceau se clôt exactement sur WAR_END
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
            log.warning(f"  ohsome réseau ({i+1}/{retries}): {e}")
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
    """Sauvegarde propre : sérialise les colonnes datetime avant export."""
    tmp = gdf.copy()
    for col in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[col]):
            tmp[col] = tmp[col].astype(str)
        elif tmp[col].dtype == object:
            # Convertir les listes/dicts en str pour GeoJSON
            try:
                tmp[col] = tmp[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict)) else x
                )
            except Exception:
                pass
    tmp.to_file(path, driver="GeoJSON")


# =============================================================================
# 1. CONTRIBUTIONS OSM — GÉOMÉTRIES RÉELLES (contributions/geometry)
# =============================================================================

def fetch_contributions(zone: str, bbox: str,
                        start: str = WAR_START,
                        end:   str = WAR_END) -> gpd.GeoDataFrame:
    """
    Récupère les géométries réelles des contributions OSM (bâtiments)
    mois par mois via contributions/geometry.

    Chaque feature retournée = un bâtiment modifié/créé/supprimé,
    avec son centroïde, sa date, et son type de contribution.

    Colonnes produites :
      geometry      : Point WGS84 (centroïde du bâtiment)
      osm_id        : identifiant OSM
      contrib_type  : creation | modification | deletion
      timestamp     : datetime UTC naïf
      date          : date seule (str YYYY-MM-DD)
      month         : période mensuelle (str YYYY-MM)
      building      : valeur du tag building
      lon / lat     : coordonnées décimales
    """
    out_file = output_path(zone, f"contributions_{zone}.geojson")
    if os.path.exists(out_file):
        log.info(f"[{zone}] contributions → cache : {out_file}")
        gdf = gpd.read_file(out_file)
        gdf["timestamp"] = pd.to_datetime(gdf.get("timestamp"), errors="coerce")
        return gdf

    log.info(f"[{zone}] Récupération contributions {start}→{end} "
             f"({len(MONTH_RANGES)} mois)…")

    month_ranges = _month_ranges(start, end)
    all_features = []

    for start_m, end_m in month_ranges:
        c_key  = f"contrib_{start_m[:7]}"
        c_file = cache_path(zone, c_key)

        if os.path.exists(c_file):
            features = json.load(open(c_file))
            log.info(f"  [{zone}] {start_m[:7]} → cache ({len(features)} ft.)")
        else:
            log.info(f"  [{zone}] {start_m[:7]} — requête ohsome…")
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

    log.info(f"[{zone}] Total features brutes : {len(all_features)}")
    if not all_features:
        log.error(f"[{zone}] Aucune contribution — vérifier bbox et période.")
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
    Récupère les suppressions OSM via Ohsome (filtrage sur @deletion côté réponse)
    """

    out_file = output_path(zone, f"deletions_{zone}.geojson")

    # 🔁 Cache
    if os.path.exists(out_file):
        log.info(f"[{zone}] deletions → cache")
        gdf = gpd.read_file(out_file)
        gdf["timestamp"] = pd.to_datetime(gdf.get("timestamp"), errors="coerce")
        return gdf

    log.info(f"[{zone}] Récupération suppressions {start}→{end}…")

    month_ranges = _month_ranges(start, end)
    all_features = []

    for start_m, end_m in month_ranges:
        c_file = cache_path(zone, f"del_{start_m[:7]}")

        # 🔁 cache mensuel
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

    # ✅ FILTRAGE DES SUPPRESSIONS ICI
    deleted_features = []
    for f in all_features:
        props = f.get("properties", {})

        if props.get("@deletion") is True:
            deleted_features.append(f)

    if not deleted_features:
        log.info(f"[{zone}] ⚠ aucune suppression trouvée")
        return gpd.GeoDataFrame()

    # 🔄 transformation vers ton format interne
    records = _parse_contribution_features(
        deleted_features,
        force_type="deletion"
    )

    if not records:
        return gpd.GeoDataFrame()

    gdf = _to_geodataframe(records)

    # 🕒 gestion timestamp
    if "timestamp" in gdf.columns:
        gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], errors="coerce")

    save_geojson(gdf, out_file)

    log.info(f"[{zone}] ✔ {len(gdf)} suppressions → {out_file}")

    return gdf


def fetch_ruins(zone: str, bbox: str) -> gpd.GeoDataFrame:
    """
    Snapshots des bâtiments tagués ruins/destroyed à des dates clés.
    Couvre toute la durée de la guerre en prenant des instantanés trimestriels.
    """
    out_file = output_path(zone, f"ruins_{zone}.geojson")
    if os.path.exists(out_file):
        log.info(f"[{zone}] ruins → cache")
        return gpd.read_file(out_file)

    # Instantanés trimestriels sur toute la guerre
    start_ts = pd.Timestamp(WAR_START)
    end_ts   = pd.Timestamp(WAR_END)
    key_dates = []
    cur = start_ts
    while cur <= end_ts:
        key_dates.append(cur.strftime("%Y-%m-%d"))
        cur += pd.DateOffset(months=3)

    log.info(f"[{zone}] Ruins — {len(key_dates)} snapshots trimestriels…")
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
# 2. ACLED — BOMBARDEMENTS UNIQUEMENT (filtre corrigé)
# =============================================================================

def load_acled_bombings(zone: str, bbox: str,
                        start: str = WAR_START,
                        end:   str = WAR_END) -> gpd.GeoDataFrame:
    """
    Charge les événements ACLED et filtre UNIQUEMENT les bombardements :
      event_type    ∋  "Explosion"    (Explosions/Remote violence)
      sub_event_type ∈  ACLED_BOMBING_SUBTYPES

    Retourne un GeoDataFrame avec colonnes normalisées :
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
        log.warning(f"ACLED introuvable : {ACLED_FILE}")
        return gpd.GeoDataFrame()

    log.info(f"[{zone}] Chargement ACLED : {ACLED_FILE}")
    gdf = gpd.read_file(ACLED_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # ── Filtre bbox ───────────────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()

    # ── Normalisation de la date ───────────────────────────────────────────────
    date_col = next(
        (c for c in gdf.columns
         if "event_date" in c or ("date" in c and "event" in c)), None
    ) or next((c for c in gdf.columns if "date" in c), None)

    if not date_col:
        log.warning(f"[{zone}] Colonne date introuvable dans ACLED")
        return gpd.GeoDataFrame()

    gdf["event_date"] = _strip_tz(
        pd.to_datetime(gdf[date_col], errors="coerce", utc=True)
    )

    # ── Filtre période ─────────────────────────────────────────────────────────
    gdf = gdf[
        (gdf["event_date"] >= pd.Timestamp(start)) &
        (gdf["event_date"] <= pd.Timestamp(end))
    ].copy()

    # ── Filtre event_type = "Explosions/Remote violence" ─────────────────────
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
        log.info(f"[{zone}] Filtre event_type 'Explosion' : "
                 f"{before} → {len(gdf)} événements")
    else:
        log.warning(f"[{zone}] Colonne event_type introuvable — filtre désactivé")

    # ── Filtre sub_event_type ─────────────────────────────────────────────────
    sub_col = next(
        (c for c in gdf.columns if "sub_event" in c), None
    )
    if sub_col:
        pattern = "|".join(ACLED_BOMBING_SUBTYPES)
        before  = len(gdf)
        gdf = gdf[
            gdf[sub_col].str.lower().str.contains(pattern, na=False)
        ].copy()
        log.info(f"[{zone}] Filtre sub_event_type bombardements : "
                 f"{before} → {len(gdf)} événements")
    else:
        log.warning(f"[{zone}] Colonne sub_event_type introuvable")

    if gdf.empty:
        log.warning(f"[{zone}] Aucun bombardement ACLED après filtrage.")
        return gpd.GeoDataFrame()

    # ── Normalisation des colonnes de sortie ──────────────────────────────────
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
    log.info(f"[{zone}] ✔ {len(gdf_out)} bombardements → {out_file}")
    return gdf_out


# =============================================================================
# 3. CROISEMENT SPATIO-TEMPOREL OSM × BOMBARDEMENTS
# =============================================================================
# SENS DE LA LECTURE (spécifique au contexte de guerre) :
#
#   On part des CONTRIBUTIONS OSM (point par point) et on cherche
#   si un bombardement a eu lieu dans les WINDOW_DAYS jours PRÉCÉDENTS
#   à moins de SPATIAL_BUFFER_M mètres.
#
#   Hypothèse : un contributeur OSM qui modifie un bâtiment peut le faire
#   en réponse à un bombardement récent dans son voisinage immédiat.
#
#   Champ produit : has_bombing_Xkm_Yd
#     1 = il y a eu un bombardement dans X km dans les Y jours précédant l'édition
#     0 = pas de bombardement proche/récent
#
#   Champs complémentaires :
#     n_bombings_Xkm_Yd  : nombre de bombardements dans la fenêtre
#     dist_nearest_m     : distance au bombardement le plus proche (mètres)
#     nearest_sub_type   : sous-type du bombardement le plus proche
#     fatalities_nearby  : total de victimes dans la fenêtre

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
        log.warning(f"[{zone}] Données manquantes pour le croisement.")
        return gpd.GeoDataFrame()

    log.info(f"[{zone}] Croisement : {len(gdf_osm):,} OSM × {len(gdf_acled):,} ACLED")

    # Projection métrique
    osm_utm   = gdf_osm.to_crs(UTM_CRS).copy()
    acled_utm = gdf_acled.to_crs(UTM_CRS).copy()

    # Dates
    osm_dates = _strip_tz(pd.to_datetime(gdf_osm["timestamp"], errors="coerce"))
    acled_dates = _strip_tz(pd.to_datetime(gdf_acled["event_date"], errors="coerce"))

    sub_type_col = "sub_event_type" if "sub_event_type" in gdf_acled.columns else None
    fatal_col    = "fatalities" if "fatalities" in gdf_acled.columns else None

    field_name   = f"has_bombing_{buffer_m//1000}km_{window_days}d"
    n_field_name = f"n_bombings_{buffer_m//1000}km_{window_days}d"

    results = []

    for i, (idx, row_osm) in enumerate(osm_utm.iterrows()):

        if i % 5000 == 0 and i > 0:
            log.info(f"  {i:,}/{len(osm_utm):,} traités…")

        edit_date = osm_dates.iloc[i]

        # Valeurs par défaut
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

        # Fenêtre temporelle (AVANT contribution)
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

        # Distances
        dists = acled_window.geometry.distance(row_osm.geometry)

        if not dists.empty:
            min_dist = float(dists.min())

            # Bombardements dans buffer
            in_buf = dists[dists <= buffer_m]
            n_in_buf = len(in_buf)

            # Plus proche événement
            if sub_type_col:
                nearest_idx = dists.idxmin()
                nearest_sub = str(acled_window.loc[nearest_idx, sub_type_col])

            # Fatalités
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

    # Fusion résultats
    res_df = pd.DataFrame(results, index=gdf_osm.index)
    gdf_out = gdf_osm.copy()

    for col in res_df.columns:
        gdf_out[col] = res_df[col].values

    save_geojson(gdf_out, out_file)

    log.info(f"[{zone}] ✔ croisement terminé → {out_file}")

    return gdf_out

# =============================================================================
# 4. SÉRIE TEMPORELLE MENSUELLE
# =============================================================================

def build_monthly_series(zone: str,
                          gdf_contrib: gpd.GeoDataFrame,
                          gdf_acled:   gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Agrège par mois :
      n_contrib_total       : toutes contributions
      n_deletions           : suppressions seulement
      n_modifications       : modifications seulement
      n_creations           : créations seulement
      n_bombings            : bombardements ACLED
      fatalities            : victimes totales
    Sauvegarde en CSV pour utilisation dans QGIS / graphiques.
    """
    out_file = output_path(zone, f"series_mensuelle_{zone}.csv")
    if os.path.exists(out_file):
        log.info(f"[{zone}] Série mensuelle → cache")
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
    log.info(f"[{zone}] ✔ Série mensuelle ({len(df)} mois) → {out_file}")
    return df


# =============================================================================
# 5. GRAPHIQUES DE SYNTHÈSE
# =============================================================================

def plot_series(zone: str, zone_name: str, df: pd.DataFrame):
    """
    Figure de synthèse pour une zone :
    - Activité OSM totale par mois (barres bleues)
    - Suppressions par mois (barres rouges)
    - Bombardements ACLED (ligne violette)
    """
    if df.empty:
        return

    os.makedirs(output_path(zone, ""), exist_ok=True)
    df = df.copy()
    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()

    ax1.bar(df["month_dt"], df["n_contrib_total"],
            width=25, color="#3498DB", alpha=0.55, label="Contributions OSM (total)")
    ax1.bar(df["month_dt"], df["n_deletions"],
            width=25, color="#E74C3C", alpha=0.85, label="Suppressions OSM")

    if df["n_bombings"].sum() > 0:
        ax2.plot(df["month_dt"], df["n_bombings"],
                 color="#8E44AD", lw=2.2, marker="o", ms=5,
                 label="Bombardements ACLED")
        ax2.fill_between(df["month_dt"], df["n_bombings"],
                         alpha=0.10, color="#8E44AD")
        ax2.set_ylabel("Bombardements ACLED / mois",
                       color="#8E44AD", fontsize=10)

    # Jalons
    ax1.axvline(pd.Timestamp("2022-02-24"), color="red", lw=1.8,
                ls="--", alpha=0.8, label="Invasion 24 fév. 2022")

    ax1.set_ylabel("Contributions / Suppressions OSM / mois",
                   color="#2C3E50", fontsize=10)
    ax1.set_title(
        f"Activité OSM × Bombardements ACLED — {zone_name}\n"
        f"Durée de la guerre : {WAR_START} → {WAR_END}",
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
# MAIN — traitement de toutes les zones configurées
# =============================================================================

def process_zone(zone: str):
    bbox, zone_name = STUDY_AREAS[zone]
    log.info("─" * 60)
    log.info(f"  ZONE : {zone_name.upper()} ({zone})")
    log.info(f"  Bbox : {bbox}")
    log.info(f"  Période : {WAR_START} → {WAR_END}")
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

    # 6. Série mensuelle
    df_series = build_monthly_series(zone, gdf_contrib, gdf_acled)

    # 7. Graphique
    if not df_series.empty:
        plot_series(zone, zone_name, df_series)

    # Résumé
    log.info(f"[{zone}] ── RÉSUMÉ ──────────────────────────────────")
    log.info(f"[{zone}] Contributions OSM    : {len(gdf_contrib):>8,}")
    log.info(f"[{zone}] dont suppressions    : {len(gdf_del):>8,}")
    log.info(f"[{zone}] Ruins/destroyed      : {len(gdf_ruins):>8,}")
    log.info(f"[{zone}] Bombardements ACLED  : {len(gdf_acled):>8,}")
    if not gdf_match.empty:
        field = f"has_bombing_{SPATIAL_BUFFER_M//1000}km_{WINDOW_DAYS}d"
        if field in gdf_match.columns:
            n_c = int(gdf_match[field].sum())
            pct = n_c / len(gdf_match) * 100
            log.info(f"[{zone}] Contributions corrélées : "
                     f"{n_c:,}/{len(gdf_match):,} ({pct:.1f}%)")
    log.info(f"[{zone}] Fichiers → {os.path.join(BASE_OUTPUT, zone)}/")


def main():
    log.info("═" * 60)
    log.info("  OSM À L'ÉPREUVE DE LA GUERRE")
    log.info(f"  Zones : {ZONES_TO_PROCESS}")
    log.info(f"  Période : {WAR_START} → {WAR_END}")
    log.info("═" * 60)

    os.makedirs(BASE_OUTPUT, exist_ok=True)
    os.makedirs(BASE_CACHE,  exist_ok=True)

    for zone in ZONES_TO_PROCESS:
        if zone not in STUDY_AREAS:
            log.error(f"Zone inconnue : {zone} — zones disponibles : "
                      f"{list(STUDY_AREAS.keys())}")
            continue
        process_zone(zone)

    log.info("═" * 60)
    log.info("  TERMINÉ — tous les GeoJSON sont prêts pour QGIS")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
