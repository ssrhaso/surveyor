#!/usr/bin/env bash
#SBATCH --job-name=2room-anchor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_anchor_%A_%a.log
# ========= TWOROOM: CLOSE THE 21pp REPRODUCTION GAP TO LeWM's PAPER =========
# LeWM report 87 on Two-Room (Fig. 6; they are the WEAKEST method there, with
# DINO-WM 100, PLDM 97, GCBC/GCIVL/GCIQL 100). Our flat reads 64-66. Until that
# gap is closed nothing about TwoRoom is interpretable, because every margin we
# have measured is against a baseline 21 points below theirs.
#
# LeWM's own config/eval/tworoom.yaml: goal_offset 25, budget 50, n_eval 50,
# horizon/receding_horizon/action_block 5/5/5, CEM 300/30/30 var 1.0, seed 42,
# NO filtering, and the goal taken from `goal_proprio`.
#
# Three differences from ours, peeled off CUMULATIVELY so the cost of each is
# attributable rather than confounded:
#   V0  ours as run                       (filters, start=final, goal_state)
#   V1  drop --require-cross-room
#   V2  also drop --eval-filter success
#   V3  also drop --episode-min holdout
#   V4  also --start random               (dataset replay, as their harness does)
#   V5  also --goal-from-proprio, n=50    == LeWM's config verbatim
#
# BAR: V5 flat >= 82, i.e. within noise of their 87. If V5 still reads ~64 the
# difference is not protocol and we look at the checkpoint next.
# Three seeds per variant; cheap, and this is the blocking measurement.
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom/ \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ID=$SLURM_ARRAY_TASK_ID
V=$((ID / 3))
SEED=$((42 + ID % 3))
N=64
ARGS=(--mode short --goal-offset 25 --eval-budget 50 --receding-horizon 5)

case $V in
  0) ARGS+=(--start final  --eval-filter success --require-cross-room --episode-min 4000) ;;
  1) ARGS+=(--start final  --eval-filter success --episode-min 4000) ;;
  2) ARGS+=(--start final  --eval-filter none    --episode-min 4000) ;;
  3) ARGS+=(--start final  --eval-filter none) ;;
  4) ARGS+=(--start random --eval-filter none) ;;
  5) ARGS+=(--start random --eval-filter none --goal-from-proprio); N=50 ;;
esac

LOG=logs/2room_anchor_v${V}_seed${SEED}.log
echo "=== anchor ablation V$V seed=$SEED n=$N : ${ARGS[*]} ==="
python -m specaccept.envs.tworoom.eval --subgoal baseline \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --num-eval "$N" --seed "$SEED" --cem-seed "$SEED" \
  "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "=== DONE V$V seed=$SEED ==="
