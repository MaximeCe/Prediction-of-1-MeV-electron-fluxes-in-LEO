# Prediction-of-1-MeV-electron-fluxes-in-LEO

Prédiction du flux d'électrons de 1 MeV en orbite basse à partir de NOAA-15 (SEM-2) et
des paramètres du vent solaire OMNI — SP519 · Space Data Processing · IPSA 2026.

## Contenu

| Fichier | Rôle |
|---|---|
| `sp519/` | Package de prétraitement (Step 1) : `config`, `io`, `clean`, `aggregate`, `qc`, `plots`, `pipeline`, `download` |
| `pre-processing-pipeline.py` | Point d'entrée en ligne de commande du Step 1 |
| `tests/` | Suite de tests `pytest` (écrite avant le code) |
| `data/noaa15_p6_hourly_Lbins.parquet` | Entrée : flux P6 NOAA-15 horaire par tranche de L (versionné) |
| `data/omni_hourly_2020_2024.parquet` | Entrée : paramètres solaires et géomagnétiques OMNI horaires (versionné) |
| `data/table_12h_2020_2024.parquet` | **Sortie du Step 1** : table 12 h (entrée du Step 2) |
| `data/table_12h_2020_2024.report.json` | Rapport de qualité du Step 1 (bilan de nettoyage, seuil p95, couverture) |
| `SP519_project.ipynb` | Notebook du projet (Step 1 : préparation et analyse ; Step 2 : machine learning) |

Les fichiers bruts NOAA-15 (~12 Go) ne sont pas versionnés : `sp519/download.py` les
re-télécharge dans `data/noaa15_raw/` (répertoire ignoré par Git) sans jamais toucher aux
tables versionnées. Leur réduction en `noaa15_p6_hourly_Lbins.parquet` est l'étape amont du
Step 1 ; tout ce qui suit part des deux parquet horaires, si bien que le pipeline complet
tourne en moins d'une seconde.

## Installation

Avec [uv](https://docs.astral.sh/uv/) (crée `.venv` et installe `requirements.txt`, relançable sans risque) :
```bash
bash setup_env.sh
```
Puis dans VS Code : *Select Kernel* → *Python Environments* → `.venv`, et exécuter le notebook de haut en bas.

## Utilisation

```bash
.venv/bin/python pre-processing-pipeline.py                 # nettoyage + table 12 h + rapport
.venv/bin/python pre-processing-pipeline.py --figures       # et les figures du Step 1 dans figures/
.venv/bin/python pre-processing-pipeline.py --l-range 3 5   # essayer une autre région de la ceinture
.venv/bin/python pre-processing-pipeline.py --download 2020 # récupérer les fichiers NOAA-15 bruts d'une année
.venv/bin/python -m pytest                                  # suite de tests
```

En Python :
```python
from sp519.config import Config
from sp519 import pipeline

result = pipeline.run(Config(l_range=(3.0, 6.0)))
result["table"]    # table longue, indexée par (Lbin, time_bin)
result["summary"]  # seuil p95, couverture, nombre d'événements
result["report"]   # trace JSON de ce qui a été nettoyé et pourquoi
```

## Le prétraitement, de bout en bout

1. **Lecture** des deux entrées horaires (`sp519.io`).
2. **Nettoyage** (`sp519.clean`) — une heure sans mesure (`n = 0`) est un trou ; un `p6_sum`
   nul est une vraie mesure « 0 coup » ; le flux horaire est `p6_sum / n` ; les heures dont
   le flux protonique (> 25 MeV, colonne `omni_p1`) dépasse le percentile `spe_percentile`
   de sa propre distribution dans la plage de L sont marquées comme contaminées.
3. **Sélection** de la région de la ceinture (plage de L justifiée, cf. ci-dessous).
4. **Agrégation** sur des intervalles de 12 h alignés sur l'epoch UTC (`sp519.aggregate`) :
   moyenne du flux, somme des échantillons bruts, nombre d'heures réellement observées,
   taux de contamination, et moyenne des paramètres OMNI sur les mêmes intervalles.
5. **Qualité** (`sp519.qc`) : couverture par tranche de L, statistiques, et seuil du
   95ᵉ percentile qui définit un « événement ».
6. **Écriture** de `data/table_12h_2020_2024.parquet` et de son rapport JSON.

Toutes les valeurs numériques du pipeline vivent dans `sp519/config.py`, chacune avec sa
justification, pour que le rapport puisse pointer un seul fichier.

## Choix justifiés

| Choix | Valeur | Justification |
|---|---|---|
| Plage de L | **3.0 – 6.0** | Sous L = 3 le canal P6 est dominé par les protons piégés (ceinture interne, anomalie de l'Atlantique sud) : le flux mesuré n'est plus celui des électrons. Au-delà de L = 6 les passages de NOAA-15 (810 km) sont trop rares. 3–6 contient le slot et tout le maximum de la ceinture externe, avec 13 à 31 mesures par heure et par tranche de 0,25 L. Cohérent avec la littérature (2.8–6.0 chez PreMevE 2.0 et Kieokaew et al. 2026). |
| Résolution | **12 h** | Imposée par l'énoncé du Step 1, et multiple des horizons +12/+24/+36/+48 h du Step 2. |
| Cible | `mep_pro_tel90_flux_p6` | Canal proton du télescope 90°, dominé par les électrons > ~1 MeV dans les ceintures ; d'où le marquage des périodes de contamination protonique. |
| Seuil d'événement | **p95** | Imposé par l'énoncé ; calculé sur la table 12 h. Pour une évaluation honnête au Step 2, le recalculer sur la seule période d'entraînement (`sp519.qc.event_threshold`). |
| Contamination protonique | percentile **99.5** du flux protonique dans la plage de L | Règle pilotée par les données plutôt qu'un seuil physique fixe : elle reste valable du minimum au maximum solaire. Le drapeau est reporté (`contaminated_frac`) et non appliqué ; `Config(drop_contaminated=True)` retire les heures marquées. |

## Résultats sur 2020–2024

- Table : 46 930 lignes = 3 610 intervalles de 12 h × 13 tranches de L (3.00 à 6.00).
- Couverture médiane des intervalles : 1.000 (minimum 0.167, pour des intervalles sous-échantillonnés).
- Seuil d'événement : **13 377 #/cm²/s/sr** (log10 = 4.13), soit 5.00 % des lignes.
- Contamination protonique : 2 775 heures marquées sur 554 955 retenues (0.5 %).

## Sources

- NOAA-15 SEM-2 / MEPED, NCEI : https://www.ncei.noaa.gov/products/poes-metop-space-environment-monitor
- OMNI2 horaire, NASA SPDF/CDAWeb : https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hourly/
