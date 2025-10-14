#!/bin/bash

# The partition we want (short=24 hours, long=7 days)
#SBATCH --partition=short
#SBATCH --time=024:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1

#SBATCH --array=0-7

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

SWEEP_ID="1ci019ty"
KEY="cef3dd8145cbd009db5a8d1e3938589896bdc25c"

source "$ENV_DIR/bin/activate"
python -m cifar10.sweep.run_sweep.py ${SWEEP_ID} ${KEY}

