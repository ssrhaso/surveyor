#!/usr/bin/env bash
#SBATCH --job-name=2room-prec2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_prec2_%A_%a.log
# ===== PRECISION EXTENSIONS II: FAIRNESS + ORACLE CEILING -> 12 SEEDS =====
# Registered in docs/tworoom_paired_prereg.md ("PRECISION EXTENSIONS II",
# 2026-07-30 evening) BEFORE submission. Anchored t=75 protocol, seeds 48-53.
# Honest exposure frozen in the prereg: the fairness re-frame trigger
# (control within 3pp of 57.03) RE-OPENS at the pooled n=12; the "at the
# oracle ceiling" sentence is re-checked against the pooled oracle.
#   IDs 0-5:  flat RH2 + DOUBLED CEM (600 samples) -- pools with 2306686
#             logs 2room_fair2x_flat_seed{48..53}.log
#   IDs 6-11: oracle s25 RH1 (Stage A ceiling arm) -- pools with 2304722
#             logs 2room_anchA_oracle_rh1_seed{48..53}.log
# ==========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ANCHORED=(--mode long --goal-offset 75 --eval-budget 150 --eval-filter none
          --episode-min 4000 --goal-from-proprio --num-eval 64)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((48 + ID % 6))

if [ "$ID" -lt 6 ]; then
  LOG=logs/2room_fair2x_flat_seed${SEED}.log
  echo "=== precision2 fairness flat RH2 + 2xCEM (600 samples) seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval --subgoal baseline \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    "${ANCHORED[@]}" --receding-horizon 2 --num-samples 600 \
    2>&1 | tee "$LOG"
else
  LOG=logs/2room_anchA_oracle_rh1_seed${SEED}.log
  echo "=== precision2 oracle s25 RH1 ceiling seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval --subgoal oracle \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    "${ANCHORED[@]}" --receding-horizon 1 --stride 25 \
    2>&1 | tee "$LOG"
fi
echo "=== DONE precision2 task $ID ==="
