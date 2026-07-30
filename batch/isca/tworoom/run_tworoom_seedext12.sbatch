#!/usr/bin/env bash
#SBATCH --job-name=2room-ext12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-29
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_ext12_%A_%a.log
# ========== SEED-SYMMETRY EXTENSIONS, PRECISION-ONLY (PRE-REGISTERED) ==========
# Registered in docs/tworoom_paired_prereg.md ("SEED-SYMMETRY EXTENSIONS",
# 2026-07-30) BEFORE submission. NO registered verdict may change: B25-1 stays
# FAILED regardless of the pooled t=25 spec mean; the 6-seed crossover verdicts
# stand at their own n; pooled 12-seed means are reported alongside (band may be
# RE-STATED wider as measured, never tightened post hoc). Log names deliberately
# match the original batteries so the existing extraction pools 12 seeds.
#   IDs 0-5:  their-protocol pair_bok_rt_rh5, seeds 48-53 (pools with 2306523)
#   IDs 6-11:  t=40 flat_rh2 seeds 48-53   IDs 12-17: t=40 spec_rh2 seeds 48-53
#   IDs 18-23: t=50 flat_rh2 seeds 48-53   IDs 24-29: t=50 spec_rh2 seeds 48-53
#   (pools with 2306577; t=60 stays 6-seed, declared tie, descriptive)
# ===============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM=gdm_tworoom_s10_gc_paired_e400v.pt
SPEC=(--subgoal specpaired --gdm-ckpt "$GDM" --gdm-steps 50 --accept-tau 0.098)
BOK=(--best-of-k 8 --bok-score goal)
RT=(--route-goal-hop 0.163)
ANCH=(--mode long --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)
THEIRS=(--mode short --start random --goal-offset 25 --eval-budget 50
        --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((48 + ID % 6))

if [ "$ID" -lt 6 ]; then
  LOG=logs/2room_theirs_pair_bok_rt_rh5_seed${SEED}.log
  echo "=== ext12 their-protocol pair_bok_rt_rh5 seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    "${THEIRS[@]}" "${SPEC[@]}" "${BOK[@]}" "${RT[@]}" --receding-horizon 5 \
    2>&1 | tee "$LOG"
else
  T=$([ "$ID" -lt 18 ] && echo 40 || echo 50)
  case $(((ID - 6) / 6 % 2)) in
    0) NAME=flat_rh2; ARGS=(--subgoal baseline --receding-horizon 2) ;;
    1) NAME=spec_rh2; ARGS=("${SPEC[@]}" --receding-horizon 2) ;;
  esac
  LOG=logs/2room_cross_${NAME}_t${T}_seed${SEED}.log
  echo "=== ext12 crossover t=$T arm=$NAME seed=$SEED ==="
  python -m specaccept.envs.tworoom.eval \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --seed "$SEED" --cem-seed "$SEED" \
    --goal-offset "$T" --eval-budget $((2 * T)) \
    "${ANCH[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
echo "=== DONE ext12 task $ID ==="
