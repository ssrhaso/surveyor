#!/usr/bin/env bash
#SBATCH --job-name=pusht-pops
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_pops_%j.log
# Fixed-population episode files for the PushT horizon battery (metadata-only
# read of the h5; quick).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
cd /lustre/home/ha676/le-wm
source "$VENV/bin/activate"
# script runs by path -> sys.path[0] is its own dir, not the repo root
export PYTHONPATH=/lustre/home/ha676/le-wm
python batch/isca/pusht/build_pusht_populations.py \
  /lustre/home/ha676/data/pusht/pusht_expert_train.h5
