#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roomwm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roomwm_%j.log
# ======== TWOROOM-DINO WORLD MODEL: the encoder-swap recovery long pole ========
# The 7/26 probe showed frozen DINOv2 restores TwoRoom's metric structure
# (equiv p50 0.876 -> 0.081, reacher-shaped gap). This trains DINO-WM's
# predictor on TwoRoom (4000 eps, frameskip 5, num_hist 3, no decoder) so the
# closed-loop recovery decision lands with the slow part done. Checkpoints
# save per epoch (model_latest.pth); walltime-bounded, resumable, and the
# whole leg is abandonable if pusht P1 fails.
# ===============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python train.py --config-name train.yaml \
  env=tworoom_dino frameskip=5 num_hist=3 has_decoder=False \
  2>&1 | tail -200
echo "=== TWOROOM WM TRAIN WALLTIME SLICE DONE ==="
