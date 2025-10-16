#!/bin/bash

# The partition we want (short=24 hours, long=7 days)
#SBATCH --partition=short
#SBATCH --time=024:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# catch an errors (i.e. git pull fails)
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
LOCK_FILE="$HOME/.ms-thesis-git-lock"

# Acquire exclusive lock to prevent concurrent git operations
exec 9>"$LOCK_FILE"
flock -x 9

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

# Release lock automatically when script ends
flock -u 9


# -----------------
# Create python env
# -----------------
ENV_DIR="$HOME/envs/ms-thesis-$SLURM_ARRAY_TASK_ID"
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade --force-reinstall pip setuptools wheel poetry --no-input --no-cache-dir
poetry install --no-interaction --no-root

which python
python -m pip list


# -------
# Run job
# -------
SWEEP_ID="mgqjtf19"
KEY=""

source "$ENV_DIR/bin/activate"
python -m mnist.run_sweep.py ${SWEEP_ID} ${KEY}

