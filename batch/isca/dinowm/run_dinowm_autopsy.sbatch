#!/usr/bin/env bash
#SBATCH --job-name=dwm-autopsy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_autopsy_%A_%a.log
# ============ DINO-WM SERVING-FAULT AUTOPSY (TIME-BOXED, PRE-REGISTERED) ============
# Registered in docs/dinowm_prereg.md ("AUTOPSY REGISTRATION 2026-07-31") BEFORE
# submission; box ends 2026-08-02 EOD. Diagnostics only -- adjudicate NOTHING
# about P1/P2/P5. Log audit already established: the old batteries were 100%
# walltime TIMEOUTs; the completed gate-test (2301459) shows the verifier NEVER
# accepts (advances 1-8 vs rejects ~230) and the gate does not fix it. Frozen
# hypothesis: drafted token grids are off-manifold for the token-level CEM cost
# while pooled conditioning/verification cannot see it.
#   0: D1 goal-serve tautology seed 99   (prediction: ~= flat 0.85-0.90)
#   1: D1 goal-serve tautology seed 100
#   2: D2 draft-and-snap seed 99         (prediction: advances >> 8, SR above
#   3: D2 draft-and-snap seed 100         the 0.05-0.20 nogate band)
#   4: D3 spec as-is seed 99             (fault reproduction control)
# Walltime sized from the measured 7.6h/run of the completed gate-test.
# ====================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python patch_cem_latentgoal.py

ID=$SLURM_ARRAY_TASK_ID
GRIDS=/lustre/home/ha676/data/dinowm/pusht_lat/grids
case $ID in
  0) MODE=goal;  SEED=99  ;;
  1) MODE=goal;  SEED=100 ;;
  2) MODE=snap;  SEED=99  ;;
  3) MODE=snap;  SEED=100 ;;
  4) MODE=draft; SEED=99  ;;
esac
NAME=${MODE}_s${SEED}
echo "=== autopsy arm=$MODE seed=$SEED (n=20, tau=0.121, k=8) ==="
ARM=spec TAU=0.121 PLAN_SEED=$SEED N_EVALS=20 SPEC_SERVE=$MODE \
  SNAP_DIR=$GRIDS SNAP_MAX=6000 \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_autopsy_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_autopsy_${NAME}.log
echo "=== AUTOPSY $NAME DONE ==="
