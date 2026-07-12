#!/usr/bin/env bash
#SBATCH --job-name=reacher-gate
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_gatejob_%A_%a.log
# Goal-gated spec-accept (--goal-gate: serve a waypoint only if it reduces
# latent goal distance, else serve the goal itself) vs the ungated battery.
# Hypothesis: parity with baseline at native t25 (ungated: 58.6 vs baseline
# 84.0) with NO regression at t100/t150 (ungated: 91.0 / 93.9).
#   tasks 0-3: t25 native protocol (episode-min 8000), seeds 42-45
#              -> compare s10spec arms (58.59) and baseline (83.99)
#   tasks 4-5: t100, max-offset-100 fixed population, seeds 42-43
#              -> compare s10spec t100 (91.02 at n=4)
#   tasks 6-7: t150, max-offset-150 fixed population, seeds 42-43
#              -> compare hz150 s10spec t150 (93.94 at n=4)
# All cells: s10 drafter, RH=2, k=8, tau=0.2 (the headline config).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"

COMMON="--subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt --goal-gate \
  --accept-tau 0.20 --gdm-steps 8 \
  --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
  --num-eval 128 --horizon 2 --receding-horizon 2"

if [ $ID -le 3 ]; then
  SEED=$((42 + ID))
  LOG=logs/reacher_gate_t25_seed${SEED}.log
  echo "=== goal-gate t25 native seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
    --dump-traces runs/reacher/traces_gate_t25_seed${SEED}.pt \
    2>&1 | tee "$LOG"
elif [ $ID -le 5 ]; then
  SEED=$((42 + ID - 4))
  EPFILE=reacher_gate.ep100.t${ID}.json
  LOG=logs/reacher_gate_t100_seed${SEED}.log
  echo "=== goal-gate t100 (max-offset-100 pop) seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    --dump-traces runs/reacher/traces_gate_t100_seed${SEED}.pt \
    2>&1 | tee "$LOG"
else
  SEED=$((42 + ID - 6))
  EPFILE=reacher_gate.ep150.t${ID}.json
  LOG=logs/reacher_gate_t150_seed${SEED}.log
  echo "=== goal-gate t150 (max-offset-150 pop) seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 150 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
    --dump-traces runs/reacher/traces_gate_t150_seed${SEED}.pt \
    2>&1 | tee "$LOG"
fi
