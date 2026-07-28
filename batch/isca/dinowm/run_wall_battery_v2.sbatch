#!/usr/bin/env bash
#SBATCH --job-name=dwm-wallbat2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_wallbat2_%A_%a.log
# ================== WALL BATTERY, CORRECTED ARM SET ==================
# Replaces 2299791, which died with DependencyNeverSatisfied because its prep
# job exited non-zero on a cosmetic QC step after saving every real artefact.
#
# The LENS ARM IS DROPPED, deliberately. P4-wall designated "the h10 lens's
# derived tau", but the lens retrained at the wall SERVING hop did not open the
# gap (width -0.131), so it has no derived tau and the original script would
# have run it at 0.20, a threshold from a different space. Inventing one to
# fill the arm would be exactly the post-hoc tuning the pre-registration
# forbids.
#
# So P4-wall is reported as measured: the prescribed mechanism was unavailable
# at the serving hop. What still runs is P3-wall, the applicability prediction,
# tested closed-loop on a task the instrument has never seen:
#   P3-wall: spec at fixed tau=0.20 UNDERPERFORMS flat.
# Arms: 0-3 flat | 4-7 spec tau=0.20. Seeds 99-102.
# ======================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
ID=$SLURM_ARRAY_TASK_ID
SEED=$((99 + ID % 4))
GDM=/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_wall_s10.pt
test -f "$GDM"
if [ $ID -le 3 ]; then ARM=flat; TAU=0; NAME=flat_s${SEED}
else                   ARM=spec; TAU=0.20; NAME=spec_t020_s${SEED}
fi
echo "=== wall battery arm=$NAME seed=$SEED ==="
ARM=$ARM TAU=$TAU PLAN_CFG=plan_wall.yaml MODEL_NAME=wall_single \
  GDM_CKPT=$GDM READOUT_CKPT= PLAN_SEED=$SEED \
  .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_wallbat2_${NAME} \
  2>&1 | tee /lustre/home/ha676/le-wm/logs/dwm_wallbat2_${NAME}.log
echo "=== WALL BATTERY $NAME DONE ==="
