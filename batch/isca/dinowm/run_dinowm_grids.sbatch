#!/usr/bin/env bash
#SBATCH --job-name=dwm-grids
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_grids_%j.log
# Full-grid re-encode of pusht_noise: (T, 257, 384) fp16 npy per episode
# (256 DINOv2 patch tokens + 1 proprio token, normalized 2-d proprio in dims
# :2). Pooled npz files already exist and are skipped; this adds drafter food
# in the exact npy format specaccept_vjepa2.train_drafter consumes verbatim.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python encode_pusht_dinowm.py
echo "=== DWM GRIDS DONE ==="
