#!/usr/bin/env bash
#SBATCH --job-name=dwm-gatetest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_gatetest_%A_%a.log
# ============ DINO-WM PUSHT: DOES THE ARRIVAL GATE FIX THE COLLAPSE? ============
# Battery 2299661 read as a total failure: flat 0.82-0.88, spec 0.00-0.06. The
# per-iteration traces say otherwise. Seed 99:
#   flat  0.06 0.16 0.32 0.44 0.48 0.56 ... 0.74 0.74   (climbs, then PLATEAUS)
#   spec  0.00 0.08 0.10 0.14 0.16 0.14 ... 0.10 0.10   (climbs, then DECLINES)
# Rising-then-falling against a plateauing baseline is the OVERSHOOT TAX: the
# drafter keeps proposing "keep going" waypoints after the agent has arrived,
# and nothing tells it to stop. Same failure as Cube and Reacher-short, where
# the arrival gate was worth +32pp. It never bit LeWM PushT because that
# dataset's episodes END at the goal; DINO-WM takes the goal from
# mid-trajectory (goal_source=dset), so arrival is not terminal.
#
# The gate was implemented twice on the LeWM stack and never ported here. It is
# now in specaccept_dinowm.py, one-way per env: once the achieved latent
# verifies against the FINAL GOAL, that env serves the goal for the rest of the
# episode at zero drafter cost. It reuses the verification test, so it adds no
# new tuned constant.
#
# PREDICTION (frozen before this ran): the gated spec arm plateaus instead of
# declining, and lands within a few points of flat. Failing that, the collapse
# is not overshoot and the DINO-WM negative stands.
#
# n_evals=20 rather than 50: every task in 2299661 hit the 12h walltime, so the
# published numbers there were mid-run, not final. Smaller n fits the budget.
# Arms: 0-1 flat | 2-3 spec no gate | 4-5 spec + gate. Seeds 99-100.
# ================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
bash /lustre/home/ha676/le-wm/batch/isca/dinowm/deploy_dinowm_gate.sh

ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 2))
GDM=/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_pusht_s25.pt
case $((ID / 2)) in
  0) ARM=flat; TAU=0;     GATE=;  NAME=flat_s${SEED} ;;
  1) ARM=spec; TAU=0.121; GATE=;  NAME=spec_nogate_s${SEED} ;;
  2) ARM=spec; TAU=0.121; GATE=1; NAME=spec_gate_s${SEED} ;;
esac

echo "=== dinowm gate test arm=$NAME seed=$SEED ==="
ARM=$ARM TAU=$TAU PLAN_CFG=plan_pusht.yaml MODEL_NAME=pusht \
  GDM_CKPT=$GDM PLAN_SEED=$SEED N_EVALS=20 MAX_ITER=12 GOAL_GATE=$GATE \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_gatetest_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_gatetest_${NAME}.log
echo "=== DONE $NAME ==="
