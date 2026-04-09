# OSM à l'épreuve de la guerre
### Analyse spatio-temporelle des contributions OpenStreetMap en Ukraine (2022–2026)

**ENSG / [anonymized] — Projet de Développement Informatique 2025-2026**  
Superviseur : Raphaël Bres  
Étudiants : [anonymized]

---

## Contexte

Ce projet analyse l'activité des contributeurs OpenStreetMap (OSM) en Ukraine depuis le début de l'invasion russe en février 2022. L'objectif est de comprendre comment une communauté de cartographie citoyenne réagit à un conflit armé de longue durée, en croisant les éditions OSM avec les données de bombardements (ACLED) et l'évolution de la ligne de front (DeepState / ISW).

Trois hypothèses sont explorées :
- **H1** — Les bombardements déclenchent des éditions OSM dans les jours qui suivent
- **H2** — L'activité OSM se concentre dans les zones libres et diminue dans les zones occupées
- **H3** — Les suppressions et tags `ruins=*` constituent un proxy de destruction physique

---

## Structure du repo

```
osm-war-ukraine/
│
├── osm_war_ukraine.py          # Analyse nationale Ukraine (4 régions, signaux OSM)
├── frontline_analysis.py       # Indicateurs OSM × ligne de front (distance, zones, buffer)
├── spatiotemporal_kyiv.py      # Série temporelle OSM × ACLED — Oblast de Kyiv
├── fetch_kyiv_osm_edits.py     # Extraction des éditions OSM pour Kyiv
│
├── analyse_spatiale_kiev.py    # Analyse exploratoire Kiev — suppressions & ruines (2022)
├── analyse_spatiale_donetsk.py # Analyse exploratoire Donetsk — suppressions & ruines (2022)
├── extraction_donees_osm_geom.py # Collecte généralisée multi-zones (Kyiv, Kharkiv, Donetsk…)
│
├── requirements.txt            # Dépendances Python
├── .gitignore
└── README.md
```

---

## Données

| Source | Usage | Accès |
|--------|-------|-------|
| [Ohsome API](https://api.ohsome.org/) | Historique complet des éditions OSM | Public |
| [ACLED](https://acleddata.com/) | Bombardements et batailles géoréférencés | Researcher tier (demande d'accès) |
| [DeepState](https://github.com/cyterat/deepstate-map-data) | Géométries quotidiennes de la ligne de front | Public |
| [ISW / Viglino](https://gist.github.com/Viglino/675e3551fb4e79d03ac0cdb1bed2677e) | Snapshots ISW de la zone occupée | Public |

> ⚠️ **ACLED** : les données brutes ne sont pas incluses dans ce repo. Elles nécessitent une demande d'accès via le [researcher tier ACLED](https://acleddata.com/). Placer le fichier `dataACLED.shp` à la racine du projet avant d'exécuter les scripts.

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/diptoxic/osm-war-ukraine.git
cd osm-war-ukraine

# Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Analyse nationale Ukraine
```bash
python osm_war_ukraine.py
```
Produit des graphiques dans `outputs_all_ukr/` : signaux OSM par région, carte des destructions, corrélation OSM × ACLED.

### Analyse ligne de front
```bash
python frontline_analysis.py
```
Produit dans `outputs/` : distance médiane contributions ↔ front, zones occupée/libre, buffer 30 km.

### Analyse Kyiv (série temporelle)
```bash
python fetch_kyiv_osm_edits.py   # extraction des données (cache local)
python spatiotemporal_kyiv.py    # visualisations
```

### Analyses exploratoires par zone (Malek Rihani)
```bash
python analyse_spatiale_kiev.py      # Kiev 2022 — suppressions & ruines
python analyse_spatiale_donetsk.py   # Donetsk 2022 — suppressions & ruines × ACLED
python extraction_donees_osm_geom.py # Collecte multi-zones configurable
```

---

## Résultats principaux

- **Signal temporel fort** : pic de contributions OSM immédiatement après le 24 février 2022, concentré sur la région de Kyiv, suivi d'une redistribution progressive
- **Zone occupée vs zone libre** : l'activité OSM chute drastiquement dans les zones sous occupation russe — l'activité cartographique suit les zones de liberté
- **Corrélation Pearson r = −0,62** (zone Donetsk) : dans les zones activement combattues, les suppressions OSM diminuent — les contributeurs ne peuvent plus cartographier
- **Tags de destruction** : les suppressions et `ruins=*` augmentent après les événements militaires majeurs, validant leur usage comme proxy de destruction

---

## Reproductibilité

Tous les scripts utilisent un **cache local JSON** pour les requêtes Ohsome API (`data/cache_ohsome/`), ce qui permet de relancer les analyses sans re-solliciter l'API. Les fichiers de cache sont exclus du repo via `.gitignore`.

---

## Perspectives

- Adaptation de la méthodologie aux routes (≈3M objets en Ukraine)
- Soumission au **State of the Map World 2026**
- Extension de la période d'analyse au-delà de 2024

---

## Références

- Goldblatt et al. (2020). *Assessing OpenStreetMap completeness for management of natural disaster*. Remote Sensing, 12(1), 118.
- Raifer et al. (2019). *OSHDB: a framework for spatio-temporal analysis of OpenStreetMap history data*. Open Geospatial Data, Software and Standards, 4(3).
- [ACLED Data](https://acleddata.com/)
- [ohsome API documentation](https://docs.ohsome.org/ohsome-api/v1/)

