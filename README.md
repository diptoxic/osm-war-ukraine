# OSM Under the Test of War
### Spatio-temporal Analysis of OpenStreetMap Contributions in Ukraine (2022–2026)

**[anonymized] — Software Development Project 2025-2026**  
Supervisor: [supervisor]  
Students: [anonymized]

---

## Context

This project analyzes the activity of OpenStreetMap (OSM) contributors in Ukraine since the beginning of the Russian invasion in February 2022. The goal is to understand how a citizen mapping community responds to a long-running armed conflict, by cross-referencing OSM edits with bombardment data (ACLED) and the evolution of the front line (DeepState / ISW).

Three hypotheses are explored:
- **H1** — Bombardments trigger OSM edits in the days that follow
- **H2** — OSM activity concentrates in free zones and decreases in occupied zones
- **H3** — Deletions and `ruins=*` tags constitute a proxy for physical destruction

---

## Repository Structure

```
osm-war-ukraine/
│
├── osm_war_ukraine.py             # National Ukraine analysis (4 regions, OSM signals)
├── frontline_analysis.py          # OSM × front line indicators (distance, zones, buffer)
├── spatiotemporal_kyiv.py         # OSM × ACLED time series — Kyiv Oblast
├── fetch_kyiv_osm_edits.py        # OSM edit extraction for Kyiv
│
├── spatial_analysis_kyiv.py       # Exploratory analysis Kyiv — deletions & ruins (2022)
├── spatial_analysis_donetsk.py    # Exploratory analysis Donetsk — deletions & ruins (2022)
├── extract_osm_geometries.py      # Generalised multi-zone collection (Kyiv, Kharkiv, Donetsk…)
│
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## Data

| Source | Usage | Access |
|--------|-------|--------|
| [Ohsome API](https://api.ohsome.org/) | Full OSM edit history | Public |
| [ACLED](https://acleddata.com/) | Geo-referenced bombardments and battles | Researcher tier (access request) |
| [DeepState](https://github.com/cyterat/deepstate-map-data) | Daily front line geometries | Public |
| [ISW / Viglino](https://gist.github.com/Viglino/675e3551fb4e79d03ac0cdb1bed2677e) | ISW snapshots of the occupied zone | Public |

> ⚠️ **ACLED**: raw data is not included in this repo. Access requires a request via the [ACLED researcher tier](https://acleddata.com/). Place the `dataACLED.shp` file at the project root before running the scripts.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/diptoxic/osm-war-ukraine.git
cd osm-war-ukraine

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### National Ukraine analysis
```bash
python osm_war_ukraine.py
```
Produces charts in `outputs_all_ukr/`: OSM signals by region, destruction map, OSM × ACLED correlation.

### Front line analysis
```bash
python frontline_analysis.py
```
Produces in `outputs/`: median distance contributions ↔ front, occupied/free zones, 30 km buffer.

### Kyiv analysis (time series)
```bash
python fetch_kyiv_osm_edits.py   # data extraction (local cache)
python spatiotemporal_kyiv.py    # visualisations
```

### Exploratory zone analyses
```bash
python spatial_analysis_kyiv.py       # Kyiv 2022 — deletions & ruins
python spatial_analysis_donetsk.py    # Donetsk 2022 — deletions & ruins × ACLED
python extract_osm_geometries.py      # Configurable multi-zone collection
```

---

## Main Results

- **Strong temporal signal**: peak of OSM contributions immediately after 24 February 2022, concentrated in the Kyiv region, followed by a gradual redistribution
- **Occupied zone vs free zone**: OSM activity drops drastically in Russian-occupied zones — mapping activity follows zones of freedom
- **Pearson correlation r = −0.62** (Donetsk zone): in actively contested zones, OSM deletions decrease — contributors can no longer map
- **Destruction tags**: deletions and `ruins=*` increase after major military events, validating their use as a destruction proxy

---

## Reproducibility

All scripts use a **local JSON cache** for Ohsome API queries (`data/cache_ohsome/`), allowing analyses to be re-run without re-querying the API. Cache files are excluded from the repo via `.gitignore`.

---

## Perspectives

- Adaptation of the methodology to roads (≈3M objects in Ukraine)
- Submission to **State of the Map World 2026**
- Extension of the analysis period beyond 2024

---

## References

- Goldblatt et al. (2020). *Assessing OpenStreetMap completeness for management of natural disaster*. Remote Sensing, 12(1), 118.
- Raifer et al. (2019). *OSHDB: a framework for spatio-temporal analysis of OpenStreetMap history data*. Open Geospatial Data, Software and Standards, 4(3).
- [ACLED Data](https://acleddata.com/)
- [ohsome API documentation](https://docs.ohsome.org/ohsome-api/v1/)
