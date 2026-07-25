#!/usr/bin/env bash
#SBATCH --job-name=dwm-anchor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_anchor_%j.log
# ================= DINO-WM PUSHT BASELINE ANCHOR (cube lesson: FIRST) =================
# Reproduce the released DINO-WM PushT planning protocol verbatim before ANY
# spec-accept work on this stack: plan_pusht.yaml = n_evals=50, seed=99,
# goal_source=dset, goal_H=5, MPC-CEM (300 samples, 30 opt steps, topk 30),
# success = pos_diff<20px AND angle<pi/9 (env.eval_state). ANCHOR BAR: our
# success_rate within noise of the paper's reported PushT number (checked at
# read-off). No spec-accept arm exists yet - drafter/gap-stat come only after
# this anchors. Runs inline via run_plan_pusht.py (their submitit launch
# hardcodes h100/qos for their cluster).
# ======================================================================================
set -euo pipefail
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export TORCH_HOME=/lustre/home/ha676/torch_cache
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
source .venv/bin/activate
python run_plan_pusht.py plan_outputs/isca_anchor_s99
echo "=== DWM ANCHOR DONE ==="
grep -h success plan_outputs/isca_anchor_s99/logs.json | tail -3 || true
