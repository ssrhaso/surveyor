#!/usr/bin/env bash
#SBATCH --job-name=reacher-oracle
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_oraclejob_%A_%a.log
# Full-power oracle arm at the native protocol: the load-bearing bound of
# sec 11.1 ("even ground-truth waypoints do not beat flat planning") currently
# rests on the n=32 smoke. Re-run at the same power as every other cited cell
# (n=128 x seeds 42-45 = 512). NATIVE ONLY: at multi-hop horizons the static
# OracleSubgoalSource measures its own staleness (non-adaptive lookup), not a
# ceiling; the audit's retracted-framing lesson; do not extend this to t>=50.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
SEED=$((42 + SLURM_ARRAY_TASK_ID))
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"
echo "=== oracle full-power: t25 native seed=$SEED n=128 ==="
python -m specaccept.envs.reacher.eval --subgoal oracle \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed $SEED --episode-min 8000 \
  --goal-offset 25 --eval-budget 50 \
  2>&1 | tee logs/reacher_oracle_t25_seed${SEED}.log
