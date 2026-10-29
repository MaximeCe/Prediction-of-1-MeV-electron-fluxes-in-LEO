#!/usr/bin/env bash
# Crée l'environnement Python du projet avec uv et installe requirements.txt.
# Usage (depuis n'importe où) :  bash setup_env.sh
# Relançable sans risque : réutilise .venv s'il existe déjà et met les paquets à jour.
set -euo pipefail

PYTHON_VERSION="3.12"
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv n'est pas installé. Installation : https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [ ! -d .venv ]; then
    echo ">> Création de .venv (Python $PYTHON_VERSION)"
    uv venv .venv --python "$PYTHON_VERSION"
else
    echo ">> .venv existe déjà, réutilisé"
fi

echo ">> Installation de requirements.txt"
uv pip install --python .venv -r requirements.txt

# Chemin de l'interpréteur : Scripts/ sous Windows, bin/ sous Linux/macOS
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

# Sous Windows 11, le contrôle d'application peut bloquer une DLL tout juste téléchargée à son
# premier chargement ("Une stratégie de contrôle d'application a bloqué ce fichier"), le temps
# de la vérifier : on réessaie donc chaque module quelques fois avant de conclure à une erreur.
echo ">> Vérification des imports (peut prendre 1-2 min la première fois)"
ECHEC=0
for m in numpy pandas matplotlib scipy h5py cdflib pyarrow sklearn ipykernel requests pytest; do
    ok=0
    for essai in 1 2 3; do
        if "$PY" -c "import $m" 2>/dev/null; then ok=1; break; fi
        sleep 5
    done
    if [ $ok -eq 1 ]; then echo "   OK  $m"; else echo "   KO  $m"; ECHEC=1; fi
done
if [ $ECHEC -eq 1 ]; then
    echo "Certains modules ne s'importent pas : relancer 'bash setup_env.sh' dans une minute."
    exit 1
fi

echo
echo "Environnement prêt : $PY"
echo "Dans VS Code : Select Kernel > Python Environments > .venv"
