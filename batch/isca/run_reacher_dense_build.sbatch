#!/usr/bin/env bash
#SBATCH --job-name=reacher-dense
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_dense_%j.log
# ISCA port of batch/run_reacher_dense_build.sbatch (Isambard). Same job, no
# apptainer: ISCA's A100 nodes are x86_64, the venv runs bare. See the
# Isambard original for the full scientific rationale.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /lustre/home/ha676/le-wm
echo "=== dense build from $H5 ==="

source "$VENV/bin/activate"
python -m specaccept.envs.reacher.build_subgoals --source local --local-dir encoder_reacher \
  --h5 "$H5" --out subgoals_reacher_dense.pt --stride 1 --device cuda --batch-size 256
