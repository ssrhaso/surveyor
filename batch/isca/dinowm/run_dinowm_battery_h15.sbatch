#!/usr/bin/env bash
#SBATCH --job-name=dwm-bat15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_bat15_%A_%a.log
# ========== PUSHT LONG-HORIZON EXTENSION goal_H=15 (pre-registered P5) ==========
# docs/dinowm_prereg.md, frozen BEFORE running: at goal_H=15 (75 raw frames,
# where the N=3 x S=25 drafted block spans the goal exactly and flat's myopia
# bites), certified spec >= flat - 2pp and the spec-vs-flat margin improves
# relative to the H5 cells. Arms: 0-3 flat, 4-7 spec tau=0.121; seeds 99-102.
# CEM horizon follows goal_H (their convention) -> ~3x compute per iteration.
# ================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 4))
if [ $ID -le 3 ]; then ARM=flat; TAU=0; else ARM=spec; TAU=0.121; fi
NAME=h15_${ARM}_s${SEED}
echo "=== battery goal_H=15 arm=$ARM tau=$TAU seed=$SEED ==="
ARM=$ARM TAU=$TAU GOAL_H=15 PLAN_SEED=$SEED \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_battery_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_bat15_${NAME}.log
echo "=== BATTERY $NAME DONE ==="
