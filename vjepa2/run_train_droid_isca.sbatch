#!/usr/bin/env bash
#SBATCH --job-name=vj2-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_train_%j.log
# ============ V-JEPA 2 TRANSPLANT: TokenGDM TRAINING (stage 2) ==============
# First drafter at video scale: 98 DROID episodes (droid_lat), S=10 (2s at
# 5fps; floor p50 at S=10 ~ 0.10 << tau=0.20 so verification headroom is
# real), N=3, goal-conditioned (manipulation default per the cube finding).
# Last 8 files held out for val loss (overfit check). Exploratory; the
# drafter's closed-loop value is judged later, this run just needs a
# converging val loss and a usable checkpoint.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
python -m specaccept_vjepa2.train_drafter \
  --data "/lustre/home/ha676/data/droid_lat/lat_*.npz" \
  --out gdm_vj2_s10.pt \
  --stride 10 --n-future 3 --epochs 150 --batch 32 --device cuda \
  2>&1 | tee ../logs/vj2_train_drafter.log
