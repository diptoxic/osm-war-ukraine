# OSM at the Test of War — Spatio-temporal Analysis, Ukraine 2022

> **Detecting Conflict-Driven OSM Mapping Activity: A Spatio-temporal Analysis, Kyiv 2022**

A research project analyzing the relationship between armed conflict events (ACLED) and OpenStreetMap editing activity in Ukraine, with a focus on Kyiv during the early phase of the 2022 invasion.

**ENSG / Université Grenoble Alpes**
Superviseur : Raphaël Bres

---

## Research hypotheses

1. **Bombings → OSM edits** — OSM building edits increase within 7 days and 500 m of a bombardment event
2. **Proximity to frontline** — OSM editing activity concentrates closer to the frontline over time
3. **Destruction signal** — building deletions and `ruins=*` tags are proxies for conflict-induced destruction

---

## Project structure

```
osm-war-ukraine/
│
├── data/                          ← place your data files here (not committed)
│   ├── dataACLED.shp              ← ACLED conflict events (+ .dbf .shx .prj)
│   └── cache_ohsome/              ← auto-generated API cache (not committed)
│
├── outputs/                       ← all generated files land here (not committed)
│
├── fetch_kyiv_osm_edits.py        ← Step 1: fetch timestamped OSM edits for Kyiv
├── spatiotemporal_kyiv.py         ← Step 2: spatio-temporal join ACLED × OSM
├── osm_war_ukraine.py             ← Step 3: national Ukraine analysis (4 regions)
├── frontline_analysis.py          ← Step 4: distance to frontline indicators
│
├── requirements.txt               ← Python dependencies
├── .gitignore
└── README.md
```

---

## Data sources

| Dataset | Source | How to get it |
|---------|--------|---------------|
| ACLED conflict events | [acleddata.com](https://acleddata.com) | Export Ukraine, all event types, 2022. Save as `dataACLED.shp` in `data/` |
| OSM building edits (timestamped) | [Ohsome API](https://api.ohsome.org) | Fetched automatically by `fetch_kyiv_osm_edits.py` |
| Frontline evolution | [ISW via Viglino Gist](https://gist.github.com/Viglino/675e3551fb4e79d03ac0cdb1bed2677e) | Fetched automatically by `frontline_analysis.py` |
| OSM buildings (current) | [Overpass Turbo](https://overpass-turbo.eu) | Run query in `overpass_query.txt`, export as GeoJSON |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourname/osm-war-ukraine.git
cd osm-war-ukraine
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n osm_war python=3.10
conda activate osm_war
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your ACLED data

Download the ACLED shapefile for Ukraine and place **all 4 files** in the `data/` folder:

```
data/
  dataACLED.shp
  dataACLED.dbf
  dataACLED.shx
  dataACLED.prj
```

---

## Run order

Run the scripts in this exact order:

### Step 1 — Fetch Kyiv OSM edits
```bash
python fetch_kyiv_osm_edits.py
```
Calls the Ohsome API and saves `outputs/kyiv_osm_edits_2022.geojson`
Runtime: ~2 minutes

### Step 2 — Spatio-temporal analysis (Kyiv)
```bash
python spatiotemporal_kyiv.py
```
Joins ACLED bombings with OSM edits (500 m radius, 7-day lag).
Produces:
- `outputs/acled_kyiv_with_osm_response.geojson`
- `outputs/matched_pairs_kyiv.csv`
- `outputs/fig_kyiv_map_response.png`
- `outputs/fig_kyiv_temporal_response.png`

### Step 3 — National Ukraine analysis
```bash
python osm_war_ukraine.py
```
Fetches OSM signals (deletions, ruins, activity) for 4 regions of Ukraine.
Results cached in `data/cache_ohsome/` — safe to interrupt and resume.
Runtime: 20–60 minutes (first run), instant on subsequent runs.

### Step 4 — Frontline distance analysis
```bash
python frontline_analysis.py
```
Computes 3 indicators: distance to frontline, occupied vs free zone activity, 30 km buffer ratio.
Runtime: 10–30 minutes (first run).

---

## Outputs

| File | Description |
|------|-------------|
| `fig_kyiv_map_response.png` | Map: bombings sized by OSM response count |
| `fig_kyiv_temporal_response.png` | Time series: bombings vs OSM edit spikes |
| `fig1_signaux_ukraine.png` | 3 OSM signals nationally over time |
| `fig2_regions.png` | Deletions by region |
| `fig3_osm_vs_acled.png` | OSM deletions vs ACLED events (Pearson r) |
| `fig4_carte_ukraine.png` | National map: deletion grid + ACLED + frontline |
| `figA_distance_front.png` | Median distance OSM edits ↔ frontline over time |
| `figB_zones_occupee_libre.png` | Edits in occupied vs free zones |
| `figC_buffer_front.png` | % edits within 30 km of frontline |
| `figD_evolution_front.png` | Frontline evolution snapshots 2022→today |

---

## QGIS visualization

Load these files into QGIS for cartographic presentation:
1. `outputs/acled_kyiv_with_osm_response.geojson` → graduated symbols on `n_osm_response`
2. `outputs/kyiv_osm_edits_2022.geojson` → heatmap (radius 1000 m, opacity 75%)
3. Basemap: XYZ → CartoDB Positron

---

## Notes

- All API calls are cached locally — scripts can be interrupted and resumed safely
- The `data/` and `outputs/` folders are excluded from git (see `.gitignore`)
- Scripts tested on Python 3.10, Windows 11 and Ubuntu 22.04
