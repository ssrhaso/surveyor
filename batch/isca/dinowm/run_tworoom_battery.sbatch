#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roombat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roombat_%A_%a.log
# ============ TWOROOM CLOSED-LOOP BATTERY (pre-registered, frozen bars) ============
# Bars frozen in docs/dinowm_prereg.md BEFORE this ran ("TwoRoom recovery,
# closed loop"). The instrument's reading at the serving configuration:
#   raw pooled gap CLOSED (equiv p90 0.096 vs hop p10 0.062, -0.034)
#   time-contrastive lens  OPEN (+0.153), derived tau = 0.497
# so it predicts a fixed tau fails here and prescribes the lens verifier.
#   P6-tworoom: spec at fixed tau=0.20 UNDERPERFORMS flat
#   P7-tworoom: lens-verified spec at tau=0.497 achieves SR >= flat - 2pp
# Arms: 0-3 spec fixed tau=0.20 | 4-7 spec lens. Seeds 99-102.
# The FLAT reference is run_tworoom_anchor.sbatch at the identical protocol and
# the identical frozen checkpoint; it is NOT re-run here and the checkpoint is
# NOT re-staged (re-staging mid-battery would break the shared-WM guarantee).
# DO NOT SUBMIT until the anchor has landed and is sane.
# ==================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm

CKPTDIR=/lustre/home/ha676/data/dinowm/checkpoints/outputs/tworoom_dino
GDM=/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_tworoom_s25.pt
LENS=/lustre/home/ha676/le-wm/readout_tworoom224_h25.pt
test -f "$CKPTDIR/FROZEN" || { echo "FAILED: anchor has not frozen the WM"; exit 1; }
test -f "$GDM"
test -f "$LENS"
[ -L "$CKPTDIR/checkpoints/model_latest.pth" ] && {
  echo "FAILED: checkpoint is a live symlink, not the frozen copy"; exit 1; }

bash deploy_tworoom_eval.sh >/dev/null
cp /lustre/home/ha676/dino_wm/plan_tworoom.yaml conf/plan_tworoom.yaml

ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 4))
if [ "$ID" -le 3 ]; then
  NAME=spec_t0.20_s${SEED}; EXTRA=""
else
  NAME=spec_lens_s${SEED};  EXTRA=$LENS
fi

echo "=== tworoom battery arm=$NAME seed=$SEED ==="
ARM=spec TAU=0.20 PLAN_CFG=plan_tworoom.yaml MODEL_NAME=tworoom_dino \
  GOAL_SOURCE=random_state N_EVALS=50 PLAN_SEED=$SEED MAX_ITER=12 \
  GDM_CKPT=$GDM READOUT_CKPT=$EXTRA \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_2roombat_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_2roombat_${NAME}.log
echo "=== TWOROOM BATTERY $NAME DONE ==="
