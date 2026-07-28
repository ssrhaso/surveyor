#!/usr/bin/env bash
#SBATCH --job-name=2room-flatsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_flatsweep_%j.log
# ============ TWOROOM STRONGEST-FLAT SWEEP (owed, and skipped until now) ============
# The paper's own write rule: "Every paper table must anchor to
# strongest-flat-per-cell, not protocol RH=5." We swept replan rate on Reacher
# (RH=2 lifted t25 from 84 to 95.7) and on PushT. We never swept it on TwoRoom,
# so the +18.8pp drafting margin at t=75 is currently measured against ONE flat
# configuration, not the best one.
#
# This sweeps receding horizon over {1, 2, 5} at both horizons, flat only. If a
# shorter replan interval lifts flat at t=75, the drafting margin shrinks by
# exactly that much and the headline has to be restated.
#
# Population sanity, already measured (probe_eval_population): neither cell is
# vacuous (1.6% / 0.0% start within the success radius) nor budget starved
# (needs p90 15 and 30 steps against budgets of 50 and 150). The 64 -> 27 drop
# across horizon is explained by the task changing: 33.6% of t=25 cells require
# crossing the wall versus 96.9% at t=75.
# ====================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

for OFF in 25 75; do
  BUD=$((2 * OFF))
  MODE=$([ "$OFF" -le 25 ] && echo short || echo long)
  for RH in 1 2 5; do
    for SEED in 42 43; do
      LOG=logs/2room_flat_rh${RH}_t${OFF}_seed${SEED}_${SLURM_JOB_ID}.log
      echo "=== flat RH=$RH t=$OFF seed=$SEED ==="
      python -m specaccept.envs.tworoom.eval --subgoal baseline \
        --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
        --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
        --start final --eval-filter success --require-cross-room --episode-min 4000 \
        --num-eval 64 --seed "$SEED" --cem-seed "$SEED" \
        --receding-horizon "$RH" \
        2>&1 | tee "$LOG" | grep -E "SR ="
    done
  done
done
echo "=== TWOROOM STRONGEST-FLAT SWEEP DONE ==="
