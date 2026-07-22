#!/usr/bin/env bash
#SBATCH --job-name=tworoom-vo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/tworoom_vo_%j.log
# ============ TWOROOM: THE CORRECTED ORACLE CEILING (EXPLORATION) =============
# The diag oracle (35.9 at t=25 vs flat 68.0) is CONTAMINATED: the static
# table advances waypoints on a schedule (~2x faster than the agent, tight
# budget) and never re-anchors; the 7/2 staleness retraction, replayed.
# Corrected arms:
#   voracle s10  = TRUE waypoints consumed like spec-accept consumes drafts
#                  (advance only on achieved-latent verification, tau=0.20).
#                  THE clean subgoal-serving ceiling.
#   oracle s25 @ t=25 = K=2 table (start+goal) => pure goal-serving through
#                  the ORACLE pathway: must ~= flat (68.0). If it does not,
#                  dataset-frame latents mismatch env-rendered goal latents
#                  and the whole oracle pathway measured the wrong quantity.
# Decision rule (frozen): voracle >= flat -> mechanics fine, drafter guilty
# (mode-averaging) -> one bounded drafter-repair attempt is justified.
# voracle < flat with sane advance stats -> decomposition genuinely
# counterproductive here -> the park stands, row unchanged.
# ==============================================================================
set -euo pipefail
cd /lustre/home/ha676/le-wm
mkdir -p logs
source .venv/bin/activate
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
H5=/lustre/home/ha676/data/tworoom/tworoom.h5

run_cell () {  # arm off seed stride
  local ARM=$1 OFF=$2 SEED=$3 STRIDE=$4
  local LOG=logs/tworoom_vo_${ARM}_s${STRIDE}_t${OFF}_seed${SEED}.log
  if [ -f "$LOG" ] && grep -q "SR =" "$LOG"; then echo "skip $LOG"; return; fi
  local MODE=short; [ "$OFF" -gt 25 ] && MODE=long
  python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --mode $MODE --goal-offset $OFF --eval-budget $((2 * OFF)) \
    --start final --eval-filter success --require-cross-room --episode-min 4000 \
    --accept-tau 0.20 --stride $STRIDE \
    --num-eval 64 --seed $SEED --cem-seed $SEED \
    2>&1 | tee "$LOG"
}

for SEED in 42 43; do
  run_cell voracle 25 $SEED 10
  run_cell voracle 75 $SEED 10
  run_cell oracle 25 $SEED 25
done
echo "=== TWOROOM VORACLE DONE ==="
