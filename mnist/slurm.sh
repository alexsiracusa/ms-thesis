#!/bin/bash

# The partition we want (short=24 hours, long=7 days)
#SBATCH --partition short
# Job duration (1 hour for test)
#SBATCH --time=01:00:00
# One node
#SBATCH -N 1
# Number of tasks on that node
#SBATCH -n 1
# Please give me a GPU
##SBATCH --gres=gpu

# Ask for memory
#SBATCH --mem=32gb

module load git
module load python/3.11.12


REPO_DIR="$HOME/ms-thesis"
git clone https://github.com/alexsiracusa/ms-thesis.git "$REPO_DIR"
cd "$REPO_DIR"


ENV_DIR="$REPO_DIR/env"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install .

SWEEP_ID="z5t9nxdv"
KEY=""

python -m mnist.run_sweep.py agent ${SWEEP_ID} ${KEY}
