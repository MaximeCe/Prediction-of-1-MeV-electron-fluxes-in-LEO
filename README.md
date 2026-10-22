# Prediction-of-1-MeV-electron-fluxes-in-LEO
Prediction of 1 MeV electron fluxes in Low Earth Orbit using NOAA-15 and solar wind data
(SP519 · Space Data Processing · IPSA 2026).

## Contenu
| Fichier | Rôle |
|---|---|
| `SP519_project.ipynb` | Notebook du projet (Step 1 : préparation et analyse ; Step 2 : machine learning) |
| `data/noaa15_p6_hourly_Lbins.parquet` | NOAA-15 P6 réduit : sommes/comptages horaires par tranche de L (versionné) |
| `data/table_12h_2020_2024.csv` | Table finale au pas de 12 h (sortie du Step 1, entrée du Step 2) |

Les fichiers bruts NOAA-15 (~12 Go) et OMNI sont re-téléchargés automatiquement si absents
(`data/noaa15_raw/`, `data/noaa15_daily/`, `data/omni/` sont ignorés par Git).

## Installation
```bash
pip install -r requirements.txt
```
Ouvrir le notebook dans VS Code, choisir l'interpréteur Python (ex. Anaconda) comme noyau, exécuter de haut en bas.

## Sources
- NOAA-15 SEM-2 / MEPED, NCEI : https://www.ncei.noaa.gov/products/poes-metop-space-environment-monitor
- OMNI2 horaire, NASA SPDF/CDAWeb : https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hourly/
