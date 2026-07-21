#!/usr/bin/env bash
#SBATCH --job-name=reacher-smoke
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_smoke_%j.log
# ISCA port of batch/run_reacher_smoke.sbatch (Isambard). Oracle + baseline at
# n=32, LeWM's reacher protocol (goal_offset 25, budget 50, qpos_match 0.05
# rad); validates the ENTIRE eval stack before any training compute.
# Success criterion: baseline SR well above 0 and oracle >= baseline.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl

source "$VENV/bin/activate"
echo '=== [1/2] ORACLE n=32 (harness + encoder + injection sanity) ==='
python -m specaccept.envs.reacher.eval --subgoal oracle \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 32 --seed 42 --episode-min 8000
echo '=== [2/2] BASELINE n=32 (flat LeWM, goal image == LeWM protocol) ==='
python -m specaccept.envs.reacher.eval --subgoal baseline \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 32 --seed 42 --episode-min 8000
