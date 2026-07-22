#!/usr/bin/env bash
#SBATCH --job-name=tworoom-dg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/tworoom_dg_%j.log
# ============ TWOROOM DIAGNOSIS (labeled EXPLORATION, not confirmation) =======
# Stage-2 verdict: PRIMARY BAR FAILED (unified 18.8/4.7 vs flat 68.0/31.2);
# secondary passed (goal-free fails as predicted). gc barely beat gf ->
# the failure is NOT (only) goal-inferability. Discriminating experiment:
#   oracle  = serve the demo's TRUE waypoints (S=10 table). If oracle ALSO
#             loses to flat -> subgoal-serving mechanics/latent geometry are
#             at fault on this env, drafter innocent. If oracle wins ->
#             drafter quality is the problem (mode-averaging at the door).
#   lerp0.5 = straight-line latent midpoint control (off-manifold probe).
# CAVEAT: OracleSubgoalSource is a STATIC table (goes stale over multi-hop
# horizons; see the 7/2 retraction); t=25 is the clean read, t=75 is a
# lower bound on the oracle ceiling.
# ==============================================================================
set -euo pipefail
cd /lustre/home/ha676/le-wm
mkdir -p logs
source .venv/bin/activate
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
H5=/lustre/home/ha676/data/tworoom/tworoom.h5

run_cell () {  # arm off seed extra...
  local ARM=$1 OFF=$2 SEED=$3; shift 3
  local LOG=logs/tworoom_diag_${ARM}_t${OFF}_seed${SEED}.log
  if [ -f "$LOG" ] && grep -q "SR =" "$LOG"; then echo "skip $LOG"; return; fi
  local MODE=short; [ "$OFF" -gt 25 ] && MODE=long
  python -m specaccept.envs.tworoom.eval --subgoal "${ARM%%[0-9.]*}" \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --mode $MODE --goal-offset $OFF --eval-budget $((2 * OFF)) \
    --start final --eval-filter success --require-cross-room --episode-min 4000 \
    --num-eval 64 --seed $SEED --cem-seed $SEED "$@" \
    2>&1 | tee "$LOG"
}

for OFF in 25 75; do for SEED in 42 43; do
  run_cell oracle $OFF $SEED --stride 10
  run_cell lerp0.5 $OFF $SEED --lerp-frac 0.5
done; done
echo "=== TWOROOM DIAG DONE ==="
