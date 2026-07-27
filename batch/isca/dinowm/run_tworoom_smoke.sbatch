#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roomsmoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roomsmoke_%j.log
# ================== TWOROOM CLOSED-LOOP SMOKE (plumbing only) ==================
# Proves the eval wrapper drives DINO-WM's MPC-CEM end to end: env registration,
# SerialVectorEnv, goal-image construction, rollout, eval_state, logging. Runs
# BOTH arms once at tiny scale so any spec-side plumbing fault surfaces here
# rather than inside a pre-registered battery.
# NOT a result: n_evals=8, max_iter=4, and the world model is still training.
# No number from this job may be quoted or written into the paper.
# ===============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm

bash deploy_tworoom_eval.sh
cp /lustre/home/ha676/dino_wm/plan_tworoom.yaml conf/plan_tworoom.yaml
bash stage_tworoom_ckpt.sh live

GDM=/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_tworoom_s25.pt
LENS=/lustre/home/ha676/le-wm/readout_tworoom224_h25.pt
test -f "$GDM"
test -f "$LENS"

echo "=== SMOKE arm=flat ==="
ARM=flat TAU=0 PLAN_CFG=plan_tworoom.yaml MODEL_NAME=tworoom_dino \
  GOAL_SOURCE=random_state N_EVALS=8 PLAN_SEED=99 MAX_ITER=4 \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/smoke_2room_flat

echo "=== SMOKE arm=spec (lens verifier, tau derived = 0.497) ==="
ARM=spec TAU=0.20 PLAN_CFG=plan_tworoom.yaml MODEL_NAME=tworoom_dino \
  GOAL_SOURCE=random_state N_EVALS=8 PLAN_SEED=99 MAX_ITER=4 \
  GDM_CKPT=$GDM READOUT_CKPT=$LENS \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/smoke_2room_speclens

echo "=== TWOROOM SMOKE DONE ==="
