#!/bin/bash

# The partition we want (short=24 hours, long=7 days)
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1

##SBATCH --array=0-3

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
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel poetry
python -m pip install .
poetry install

which python
python -m pip list

SWEEP_ID="d77uvkc9"
KEY=""

source "$ENV_DIR/bin/activate"

#for ((i = 0; i < 8; i++)); do
#  srun --mem-per-cpu=4G python -m mnist.run_sweep.py ${SWEEP_ID} ${KEY}
#done
python -m mnist.run_sweep.py ${SWEEP_ID} ${KEY}

