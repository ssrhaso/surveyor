#!/usr/bin/env bash
#SBATCH --job-name=2room-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-47
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_gate_%A_%a.log
# ============ TWOROOM: THE CEILING, AND THE GATE WE NEVER WIRED ============
# Two things this answers that nothing so far has.
#
# 1. IS THERE ANY HEADROOM? Every TwoRoom comparison to date has been spec
#    versus flat with no idea what the achievable maximum is. If the oracle
#    sits at flat, the environment is saturated and no method can win, and we
#    should stop. oracle serves TRUE demo waypoints on a schedule; voracle
#    serves them the way spec-accept consumes drafts, advancing only when the
#    achieved latent verifies. voracle is the clean subgoal-serving ceiling.
#
# 2. DOES THE ARRIVAL GATE HELP? --goal-gate existed only in the cube and
#    reacher drivers and was NEVER available on TwoRoom. On Cube it was the
#    difference between a loss and a win: the overshoot tax alone accounted for
#    +32pp of that margin. TwoRoom's protocol gives eval_budget = 2 x
#    goal_offset, i.e. twice the time the goal needs, which is exactly the
#    condition under which a drafter walks the agent back off a reached target.
#    The gate reuses the accept test, so it introduces no new constant.
#
# LeWM's own config (config/eval/tworoom.yaml) is goal_offset 25, budget 50,
# horizon/receding_horizon/action_block 5/5/5, CEM 300/30/30 var 1.0 - which our
# protocol already matches. t=75 is our extension, as on every other env.
#
# Six seeds throughout: two-seed margins have twice evaporated under more seeds.
# Arms x t{25,75} x seeds{42..47}. Reference flat comes from 2300264/2301412.
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
GDM576=gdm_tworoom_s10_gc_paired_e400v.pt
test -f "$GDM576"

ID=$SLURM_ARRAY_TASK_ID
OFF=$([ $((ID / 24)) -eq 0 ] && echo 25 || echo 75)
REST=$((ID % 24))
SEED=$((42 + REST % 6))
BUD=$((2 * OFF))
MODE=$([ "$OFF" -le 25 ] && echo short || echo long)
# RH at each horizon = flat's own strongest, so the gate is tested against the
# hardest baseline rather than a convenient one
RH=$([ "$OFF" -le 25 ] && echo 2 || echo 1)

BANK=subgoals_tworoom_s10_paired.pt
SPEC=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.098)
case $((REST / 6)) in
  # the ceiling: if oracle sits at flat, TwoRoom is saturated and we stop
  0) ARM=oracle;     NAME=oracle;   EXTRA=(--stride 10) ;;
  1) ARM=voracle;    NAME=voracle;  EXTRA=(--stride 10 --accept-tau 0.20) ;;
  # snapping alone, to isolate reachability from the other two mechanisms
  2) ARM=specpaired; NAME=snap;     EXTRA=("${SPEC[@]}" --snap-bank "$BANK") ;;
  # flagship: reachable waypoints + progress constraint + arrival gate
  3) ARM=specpaired; NAME=flagship; EXTRA=("${SPEC[@]}" --snap-bank "$BANK" --snap-progress --goal-gate) ;;
esac

LOG=logs/2room_gate_${NAME}_t${OFF}_rh${RH}_seed${SEED}.log
echo "=== arm=$NAME t=$OFF RH=$RH seed=$SEED ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
  --start final --eval-filter success --require-cross-room --episode-min 4000 \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon "$RH" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE arm=$NAME t=$OFF RH=$RH seed=$SEED ==="
