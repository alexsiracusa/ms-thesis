#!/bin/bash

# The partition we want (short=24 hours, long=7 days)
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1


module load git
module load python/3.11.12


REPO_DIR="$HOME/ms-thesis"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull
else
    git clone https://github.com/alexsiracusa/ms-thesis.git "$REPO_DIR"
    cd "$REPO_DIR"
fi


ENV_DIR="$REPO_DIR/env"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install .

SWEEP_ID="z5t9nxdv"
KEY=""

python -m mnist.run_sweep.py agent ${SWEEP_ID} ${KEY}
