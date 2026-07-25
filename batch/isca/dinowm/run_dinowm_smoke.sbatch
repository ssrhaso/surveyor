#!/usr/bin/env bash
#SBATCH --job-name=dwm-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_smoke_%j.log
# DINO-WM PushT smoke: 2 evals through the full inline pipeline (ckpt load,
# dset goals, SubprocVectorEnv, MPC-CEM, eval_state). Gates the anchor run.
set -euo pipefail
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export TORCH_HOME=/lustre/home/ha676/torch_cache
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
source .venv/bin/activate
N_EVALS=2 python run_plan_pusht.py plan_outputs/isca_smoke
echo "=== DWM SMOKE OK ==="
