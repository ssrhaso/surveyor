#!/usr/bin/env bash
#SBATCH --job-name=reacher-rh2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_rh2job_%A_%a.log
# Replan-rate control: flat LeWM baseline (goal image, no drafter) at RH=2
# (replan every 10 steps) instead of the LeWM-protocol RH=5 (every 25).
# Disambiguates the goal-gate battery: gated spec-accept behaved ~= baseline
# at t100/t150 but scored 91.6 at t25 vs protocol-baseline 84.0; if THIS
# control also scores ~91 at t25, the gate's t25 "win" was just replan
# frequency and the gate criterion (greedy latent progress) is refuted in
# both directions.
#   tasks 0-3: t25 native (episode-min 8000), seeds 42-45
#   tasks 4-5: t100, max-offset-100 fixed population, seeds 42-43
#   tasks 6-7: t150, max-offset-150 fixed population, seeds 42-43
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"

COMMON="--subgoal baseline --accept-tau 0.20 --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
  --num-eval 128 --horizon 2 --receding-horizon 2"

if [ $ID -le 3 ]; then
  SEED=$((42 + ID))
  echo "=== baseline RH=2 t25 native seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
    2>&1 | tee logs/reacher_rh2_t25_seed${SEED}.log
elif [ $ID -le 5 ]; then
  SEED=$((42 + ID - 4))
  EPFILE=reacher_rh2.ep100.t${ID}.json
  echo "=== baseline RH=2 t100 seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    2>&1 | tee logs/reacher_rh2_t100_seed${SEED}.log
else
  SEED=$((42 + ID - 6))
  EPFILE=reacher_rh2.ep150.t${ID}.json
  echo "=== baseline RH=2 t150 seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 150 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
    2>&1 | tee logs/reacher_rh2_t150_seed${SEED}.log
fi
