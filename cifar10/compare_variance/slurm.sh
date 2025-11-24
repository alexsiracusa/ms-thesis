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

REPO_DIR="$HOME/ms-thesis"
ENV_DIR="$HOME/envs/ms-thesis"
SWEEP_ID="jvdv4c6g"
KEY=""

cd "$REPO_DIR"
source "$ENV_DIR/bin/activate"
python -m cifar10.compare_variance.run_sweep.py ${SWEEP_ID} ${KEY}

