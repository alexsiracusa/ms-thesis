#!/bin/bash

set -euo pipefail
trap 'echo "❌ Error on line $LINENO. Exiting..." >&2' ERR

# Load slurm modules
module load git
module load python/3.11.12

# --------------
# Clone git repo
# --------------
REPO_URL="https://github.com/alexsiracusa/ms-thesis.git"
REPO_DIR="$HOME/ms-thesis"

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    echo "Updating existing repository..."
    git fetch origin main
    git reset --hard origin/main
else
    echo "Cloning fresh repository..."
    git clone --depth=1 "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi


# -----------------
# Create python env
# -----------------
ENV_DIR="$HOME/envs/ms-thesis"
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade --force-reinstall pip setuptools wheel poetry --no-input --no-cache-dir
poetry install --no-interaction --no-root

which python
python -m pip list
