#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roomwm-r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roomwm_resume_%j.log
# ============ TWOROOM WM: RESUME SLICE (chain behind the first slice) ============
# train.py resumes automatically from $saved_folder/checkpoints/model_latest.pth
# (init_models), and saved_folder IS the hydra run dir -- which hydra re-stamps
# per launch. So the run dir must be PINNED to the original run or each slice
# silently restarts from scratch at epoch 0.
#
# Rate as measured on the first slice: ~1.5 h/epoch (the frozen DINOv2 encoder
# runs on every sample; epochs=100 would need ~6 days). Each slice adds ~16
# epochs. Chain as many as the deadline allows; the checkpoint is usable at any
# epoch and stage_tworoom_ckpt.sh records which epoch a run served.
# ================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm

RUN_DIR=${RUN_DIR:-/lustre/home/ha676/dino_wm/outputs/2026-07-27/10-04-16}
test -f "$RUN_DIR/checkpoints/model_latest.pth" || {
  echo "FAILED: no checkpoint to resume at $RUN_DIR"; exit 1; }
.venv/bin/python - "$RUN_DIR/checkpoints/model_latest.pth" <<'PY'
import sys
import torch
print("[resume] from epoch:", torch.load(sys.argv[1], map_location="cpu").get("epoch"))
PY

.venv/bin/python train.py --config-name train.yaml \
  env=tworoom_dino frameskip=5 num_hist=3 has_decoder=False \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tail -200
echo "=== TWOROOM WM RESUME SLICE DONE ==="
