#!/usr/bin/env bash
#SBATCH --job-name=dwm-wallbat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_wallbat_%A_%a.log
# ============== WALL CLOSED-LOOP BATTERY (pre-registered, docs/dinowm_prereg.md) ==============
# The instrument's wall reading: gap CLOSED at fixed tau (tails overlap,
# reacher shape) but the LENS opens it. Frozen bars BEFORE this ran:
#   P3-wall: spec at fixed tau=0.20 UNDERPERFORMS flat (confirming the
#            applicability prediction on a never-run task)
#   P4-wall: lens-verified spec >= flat - 2pp (the prescribed mechanism works)
# Arms: 0-3 flat | 4-7 spec fixed tau=0.20 | 8-11 spec lens (tau from
# readout_dinowm_wall_h10.pt's derived value). Seeds 99-102, n_evals=50,
# max_iter=12, protocol identical to the wall anchor.
# ==============================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 4))
GDM=/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_wall_s10.pt
LENS=/lustre/home/ha676/le-wm/readout_dinowm_wall_h10.pt
test -f "$GDM"
if   [ $ID -le 3 ]; then ARM=flat; TAU=0; EXTRA=""
elif [ $ID -le 7 ]; then ARM=spec; TAU=0.20; EXTRA=""
else                     ARM=spec; TAU=0.20; EXTRA=$LENS; test -f "$LENS"
fi
NAME=${ARM}_$([ -n "$EXTRA" ] && echo lens || echo t${TAU})_s${SEED}
echo "=== wall battery arm=$NAME seed=$SEED ==="
ARM=$ARM TAU=$TAU PLAN_CFG=plan_wall.yaml MODEL_NAME=wall_single \
  GDM_CKPT=$GDM READOUT_CKPT=$EXTRA PLAN_SEED=$SEED \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_wallbat_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_wallbat_${NAME}.log
echo "=== WALL BATTERY $NAME DONE ==="
