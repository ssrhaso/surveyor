#!/usr/bin/env bash
#SBATCH --job-name=dwm-battery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_battery_%A_%a.log
# ============ DINO-WM PUSHT CLOSED-LOOP BATTERY (prospective test #4) ============
# The pre-registered arms (docs/dinowm_prereg.md, FROZEN 2026-07-26 BEFORE this
# ran): 0-3 flat | 4-7 spec tau=0.121 (in-gap, P1: SR >= flat - 2pp)
#       | 8-11 spec tau=0.20 (ABOVE gap, P2: degenerate-accept mechanics,
#       rejects ~0). Seeds 99-102 (their seed convention), n_evals=50,
#       max_iter=12, protocol identical to the anchor in every other respect.
# =================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python patch_cem_latentgoal.py
ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 4))
if   [ $ID -le 3 ]; then ARM=flat; TAU=0
elif [ $ID -le 7 ]; then ARM=spec; TAU=0.121
else                     ARM=spec; TAU=0.20
fi
NAME=${ARM}_t${TAU}_s${SEED}
echo "=== battery arm=$ARM tau=$TAU seed=$SEED ==="
ARM=$ARM TAU=$TAU PLAN_SEED=$SEED \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_battery_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_battery_${NAME}.log
echo "=== BATTERY $NAME DONE ==="
