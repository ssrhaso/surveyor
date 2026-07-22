#!/usr/bin/env bash
#SBATCH --job-name=tworoom-s2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/tworoom_s2_%j.log
# ============ TWOROOM STAGE 2: THE INTERVENTION EXPERIMENT (PRE-REGISTERED) =
# Anchor (stage 1, banked): flat cross-room SR = 64.1 (t=25) / 27.3 (t=75) --
# flat collapses at range, ~37pp headroom at t=75.
# CLAIM UNDER TEST: TwoRoom violates ONE spec-accept scope condition (goal
# not inferable from the CURRENT observation); supplying the goal (which the
# eval protocol renders) restores applicability. Arms, all tau=0.20 frozen,
# S=10, N=3, gdm-steps 8 (k underived on TwoRoom; stated caveat, any k
# sweep afterward is labeled exploration):
#   flat        seeds 44-45 (completing the 4-seed anchor; 42-43 banked)
#   spec-gf     goal-FREE drafter: the scope-condition row.
#     PREDICTION (frozen): fails; SR <= flat at both t.
#   spec-gc     goal-conditioned drafter, pure spec-accept.
#   unified     goal-conditioned + c*-retire (the certified method).
# BARS (frozen NOW; means over seeds 42-45, n=64/cell):
#   PRIMARY:   unified >= flat at BOTH t; at t=75 unified >= flat + 10pp.
#   SECONDARY: spec-gf <= flat at both t (the intervention's control leg).
# Report as-is, no amendment on failure.
# ==============================================================================
set -euo pipefail
cd /lustre/home/ha676/le-wm
mkdir -p logs
source .venv/bin/activate
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
H5=/lustre/home/ha676/data/tworoom/tworoom.h5

echo "=== stage 2a: train goal-FREE drafter (control leg) ==="
if [ ! -f gdm_tworoom_s10_gf.pt ]; then
  python -m specaccept.train_drafter \
    --subgoals subgoals_tworoom_s10.pt --out gdm_tworoom_s10_gf.pt \
    --mask 5 --device cuda --seed 42 \
    2>&1 | tee logs/tworoom_train_gf.log
fi

run_cell () {  # arm ckpt off seed extra...
  local ARM=$1 CKPT=$2 OFF=$3 SEED=$4; shift 4
  local LOG=logs/tworoom_${ARM}_t${OFF}_seed${SEED}.log
  if [ -f "$LOG" ] && grep -q "SR =" "$LOG"; then echo "skip $LOG"; return; fi
  local MODE=short; [ "$OFF" -gt 25 ] && MODE=long
  python -m specaccept.envs.tworoom.eval \
    --subgoal "${ARM%%-*}" ${CKPT:+--gdm-ckpt $CKPT} \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --mode $MODE --goal-offset $OFF --eval-budget $((2 * OFF)) \
    --start final --eval-filter success --require-cross-room --episode-min 4000 \
    --gdm-steps 8 --accept-tau 0.20 \
    --num-eval 64 --seed $SEED --cem-seed $SEED "$@" \
    2>&1 | tee "$LOG"
}

echo "=== stage 2b: fill flat anchor seeds 44-45 ==="
for OFF in 25 75; do for SEED in 44 45; do
  run_cell baseline "" $OFF $SEED
done; done

echo "=== stage 2c: goal-free spec (predicted failure) ==="
for OFF in 25 75; do for SEED in 42 43 44 45; do
  run_cell specaccept-gf gdm_tworoom_s10_gf.pt $OFF $SEED
done; done

echo "=== stage 2d: goal-cond spec (pure) ==="
for OFF in 25 75; do for SEED in 42 43 44 45; do
  run_cell specaccept-gc gdm_tworoom_s10_gc.pt $OFF $SEED
done; done

echo "=== stage 2e: unified certified (the intervention arm) ==="
for OFF in 25 75; do for SEED in 42 43 44 45; do
  run_cell unified-gc gdm_tworoom_s10_gc.pt $OFF $SEED
done; done
echo "=== TWOROOM STAGE 2 DONE ==="
