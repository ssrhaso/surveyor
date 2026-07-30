#!/usr/bin/env bash
#SBATCH --job-name=2room-cross
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-83
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_cross_%A_%a.log
# ========= TWOROOM CROSSOVER SWEEP + 12-SEED SYMMETRY FILLS =========
# Registered in docs/tworoom_paired_prereg.md before submission (descriptive;
# frozen prediction: flat still ahead at t=40, spec ahead by t in [50,60],
# crossover where the goal exits the ~25-step planning window + one stride).
#   IDs 0-71: t in {40,50,60} x {flat_rh2, flat_rh5, spec_rh2, spec_rh5} x 6 seeds
#   IDs 72-77: bok-arm extension seeds 48-53 at t=75 RH2 (pools to 12 seeds)
#   IDs 78-83: their-protocol flat RH5 seeds 48-53 (pools to 12 seeds)
# =====================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM=gdm_tworoom_s10_gc_paired_e400v.pt
SPEC=(--subgoal specpaired --gdm-ckpt "$GDM" --gdm-steps 50 --accept-tau 0.098)
ANCH=(--mode long --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)

ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -lt 72 ]; then
  SEED=$((42 + ID % 6))
  TS=(40 50 60); T=${TS[$((ID / 24))]}
  case $(((ID / 6) % 4)) in
    0) NAME=flat_rh2; ARGS=(--subgoal baseline --receding-horizon 2) ;;
    1) NAME=flat_rh5; ARGS=(--subgoal baseline --receding-horizon 5) ;;
    2) NAME=spec_rh2; ARGS=("${SPEC[@]}" --receding-horizon 2) ;;
    3) NAME=spec_rh5; ARGS=("${SPEC[@]}" --receding-horizon 5) ;;
  esac
  LOG=logs/2room_cross_${NAME}_t${T}_seed${SEED}.log
  echo "=== crossover t=$T arm=$NAME seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    --goal-offset "$T" --eval-budget $((2 * T)) \
    "${ANCH[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG"
elif [ "$ID" -lt 78 ]; then
  SEED=$((48 + ID - 72))
  LOG=logs/2room_cross_bokext_t75_seed${SEED}.log
  echo "=== bok extension t=75 seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    --goal-offset 75 --eval-budget 150 --receding-horizon 2 \
    "${ANCH[@]}" "${SPEC[@]}" --best-of-k 8 --bok-score goal \
    2>&1 | tee "$LOG"
else
  SEED=$((48 + ID - 78))
  LOG=logs/2room_cross_theirsflat_seed${SEED}.log
  echo "=== their-protocol flat RH5 extension seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval --subgoal baseline \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    --mode short --start random --goal-offset 25 --eval-budget 50 \
    --eval-filter none --episode-min 4000 --goal-from-proprio \
    --num-eval 64 --receding-horizon 5 \
    2>&1 | tee "$LOG"
fi
echo "=== DONE task $ID ==="
