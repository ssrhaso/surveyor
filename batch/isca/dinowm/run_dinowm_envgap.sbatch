#!/usr/bin/env bash
#SBATCH --job-name=dwm-envgap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_envgap_%j.log
# ====== DINO-WM EXTRA ENVS: ENCODE + GAP STATISTIC (more prospective rows) ======
# MODEL_NAME in {point_maze, wall_single} (sbatch --export). Same instrument
# protocol as pusht: encode the released model's latents, gap stat on the
# valid split at hops {5,25,50}, tau=0.20 reference. Each result = one more
# PRE-REGISTERED applicability prediction on a task we have never run
# closed-loop; closed-loop confirmation optional per prediction + env deps.
# =================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
: "${MODEL_NAME:?need MODEL_NAME}"
cd /lustre/home/ha676/dino_wm
MODEL_NAME=$MODEL_NAME .venv/bin/python encode_dinowm_env.py
echo "=== gap stat (le-wm venv) ==="
cd /lustre/home/ha676/le-wm
for H in 5 25 50; do
  .venv/bin/python -m specaccept.probes.gap_stat --name ${MODEL_NAME}_h${H} \
    --lat-glob "/lustre/home/ha676/data/dinowm/${MODEL_NAME}_lat/lat_${MODEL_NAME}_valid_*.npz" \
    --equiv-stride 1 --hop-stride ${H} --tau 0.20 \
    --json-out gap_dinowm_${MODEL_NAME}_h${H}.json
done
echo "=== ${MODEL_NAME} ENCODE+GAP DONE ==="
