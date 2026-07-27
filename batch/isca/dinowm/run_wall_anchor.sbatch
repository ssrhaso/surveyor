#!/usr/bin/env bash
#SBATCH --job-name=dwm-wallanch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_wallanch_%j.log
# WALL flat baseline anchor: their released plan_wall protocol verbatim
# (n_evals=50, seed=99, goal_source=random_state, goal_H=5, MPC-CEM 300x10)
# + the same max_iter=12 episode-budget cap as every arm on this stack.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
ARM=flat PLAN_CFG=plan_wall.yaml MODEL_NAME=wall_single \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_wall_anchor_s99
echo "=== WALL ANCHOR DONE ==="
