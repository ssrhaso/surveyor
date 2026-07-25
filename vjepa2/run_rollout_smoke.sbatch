#!/usr/bin/env bash
#SBATCH --job-name=vj2-ra-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_ra_smoke_%j.log
# Smoke gate for the overnight rollout-agreement fleet: probe_rollout_agreement
# has never run on GPU. 1 episode, 1 anchor, budget 25, ALL FOUR ARMS (validates
# the route/tcal path too). The three arrays depend afterok on this job -
# a code bug costs 40 min, not a fleet-night.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1 LC_ALL=C
cd /lustre/home/ha676/le-wm/vjepa2
source ../.venv/bin/activate
LAT=/lustre/home/ha676/data/droid2k_lat
EP=$(ls $LAT/lat_droid_ep*[0-9].npy | sort | tail -100 | sed -n '29p')
python -m specaccept_vjepa2.probe_rollout_agreement \
  --ckpt gdm_vj2_s10.pt --episodes "$EP" \
  --goal-offset 32 --budget 25 --anchors-per-ep 1 --max-eps 1 \
  --arms flat,spec,lerp,route \
  --out ra_smoke.npz --device cuda
echo "=== RA SMOKE OK ==="
