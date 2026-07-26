#!/usr/bin/env bash
#SBATCH --job-name=dwm-encgap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_encgap_%j.log
# ============ DINO-WM PUSHT: ENCODE LATENTS + GAP STATISTIC (prospective #4) ============
# Stage 1: encode pusht_noise through the trained model's own encode path
# (frozen DINOv2 ViT-S/14 patch tokens, pooled per frame) - drafter food AND
# gap-stat input. Stage 2: the verification-gap statistic on the VALID split
# at hops {5, 25, 50} raw frames (1 model step = 5 frames; goal_H=5 = 25).
# The resulting gap_dinowm_pusht_h*.json values feed the PRE-REGISTRATION
# (docs/dinowm_prereg.md in le-wm): the instrument's applicability + mechanism
# prediction gets frozen from these numbers BEFORE any closed-loop spec run.
# ========================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python encode_pusht_dinowm.py

echo "=== stage 2: gap statistic (le-wm venv) ==="
cd /lustre/home/ha676/le-wm
for H in 5 25 50; do
  .venv/bin/python -m specaccept.probes.gap_stat --name dinowm_pusht_h${H} \
    --lat-glob "/lustre/home/ha676/data/dinowm/pusht_lat/lat_pusht_valid_*.npz" \
    --equiv-stride 1 --hop-stride ${H} --tau 0.20 \
    --json-out gap_dinowm_pusht_h${H}.json
done
echo "=== DWM ENCODE+GAP DONE ==="
