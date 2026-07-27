#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roomanch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roomanch_%A_%a.log
# ==================== TWOROOM FLAT ANCHOR (run this FIRST) ====================
# The cube lesson, applied: nothing about TwoRoom may be concluded until flat
# planning is anchored on this substrate. This is the ONLY arm in this job.
#
# Protocol (becomes the certified TwoRoom protocol if the anchor is sane):
#   goal_source=random_state, init/goal in opposite rooms, goal_H=5,
#   frameskip=5, MPC-CEM 300 samples x 10 opt steps, n_taken_actions=5,
#   max_iter=12 (= 300 env steps; the data's door-routed path needs ~38 env
#   steps at p50 and ~54 at p90, so the budget is ~6x the requirement and
#   cannot starve the planner), n_evals=50, seeds 99-102.
#   success = final agent within 16 px of the goal (TwoRoomEnv's own criterion).
#
# Task 0 freezes the world model so every arm -- anchor and battery alike --
# serves one identical checkpoint; the other tasks wait for that file.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm

FROZEN=/lustre/home/ha676/data/dinowm/checkpoints/outputs/tworoom_dino/checkpoints/model_latest.pth
ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID))

bash deploy_tworoom_eval.sh >/dev/null
cp /lustre/home/ha676/dino_wm/plan_tworoom.yaml conf/plan_tworoom.yaml

if [ "$ID" -eq 0 ]; then
  bash stage_tworoom_ckpt.sh freeze
  touch /lustre/home/ha676/data/dinowm/checkpoints/outputs/tworoom_dino/FROZEN
else
  for _ in $(seq 1 120); do
    [ -f /lustre/home/ha676/data/dinowm/checkpoints/outputs/tworoom_dino/FROZEN ] && break
    sleep 10
  done
fi
test -f "$FROZEN"
.venv/bin/python - "$FROZEN" <<'PY'
import sys
import torch
print("[anchor] serving WM epoch:", torch.load(sys.argv[1], map_location="cpu").get("epoch"))
PY

echo "=== tworoom anchor arm=flat seed=$SEED ==="
ARM=flat TAU=0 PLAN_CFG=plan_tworoom.yaml MODEL_NAME=tworoom_dino \
  GOAL_SOURCE=random_state N_EVALS=50 PLAN_SEED=$SEED MAX_ITER=12 \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_2roomanch_flat_s${SEED} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_2roomanch_flat_s${SEED}.log
echo "=== TWOROOM ANCHOR flat seed=$SEED DONE ==="
