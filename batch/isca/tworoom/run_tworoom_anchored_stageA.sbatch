#!/usr/bin/env bash
#SBATCH --job-name=2room-anchA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_anchA_%A_%a.log
# ========== TWOROOM STAGE A: FLAT SWEEP + CEILING AT THE ANCHORED PROTOCOL ==========
# The anchor ablation (2304691) closed the 21pp gap: V5 (LeWM's config verbatim)
# reads 85.33 vs their 87, and the cost of every protocol difference is measured
# (success filter ~14.6pp dominates). The horizon-extension result must now be
# re-measured at the ANCHORED protocol: unfiltered population, holdout kept,
# goal_proprio (their goal source; also the coherent one on unfiltered episodes).
#
# This stage SETS the bar for Stage B (docs/tworoom_paired_prereg.md, 2026-07-29
# amendment): flat swept over RH {1,2,5} x 6 seeds, plus the oracle ceiling row.
# Stage B gate: oracle >= strongest flat + 3pp, else the anchored t=75 population
# has no decomposition headroom and Stage B cannot be read as a win.
# ====================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 6))
ANCHORED=(--mode long --goal-offset 75 --eval-budget 150 --eval-filter none
          --episode-min 4000 --goal-from-proprio --num-eval 64)

case $((ID / 6)) in
  0) ARM=baseline; NAME=flat_rh1;   EXTRA=(--receding-horizon 1) ;;
  1) ARM=baseline; NAME=flat_rh2;   EXTRA=(--receding-horizon 2) ;;
  2) ARM=baseline; NAME=flat_rh5;   EXTRA=(--receding-horizon 5) ;;
  3) ARM=oracle;   NAME=oracle_rh1; EXTRA=(--receding-horizon 1 --stride 25) ;;
esac

LOG=logs/2room_anchA_${NAME}_seed${SEED}.log
echo "=== Stage A arm=$NAME seed=$SEED : anchored protocol ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  "${ANCHORED[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE $NAME seed=$SEED ==="
