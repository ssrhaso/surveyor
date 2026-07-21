#!/usr/bin/env bash
#SBATCH --job-name=gatev4c-route
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev4c_route_%j.log
# ================= GATE V4 CONFIRM: ROUTING PASS (PRE-REGISTERED) ===========
# Written BEFORE any confirmation data. The exploratory v4 run (pop seed 888,
# jobs 2274090/2274091) gave composite 98.05/97.07/93.75/93.55 vs LeWM-best
# 97.07/96.88/83.79/75.20 -> beat both flat arms at every t. This chain is its
# frozen-bar confirmation on builder seed 999 (never used anywhere).
# Rule FROZEN as in the exploratory run: c*_2<=tau=0.20 -> flat2; elif
# c*_5<=tau -> flat5; else spec. Routing cem-seed 42, one decision per
# (episode, t), episode-start info only. tau and the rule are frozen; failure
# is reported as-is, no post-hoc amendment.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /lustre/home/ha676/le-wm
mkdir -p logs Results
source "$VENV/bin/activate"
export MUJOCO_GL=egl
python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
  --out reacher_gatev4c.ep100.s999.json --n 128 --max-offset 100 \
  --episode-min 8000 --seed 999
python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
  --out reacher_gatev4c.ep150.s999.json --n 128 --max-offset 150 \
  --episode-min 8000 --seed 999
python -m specaccept.probes.route_horizon \
  --episodes-file reacher_gatev4c.ep100.s999.json --offsets 25 50 100 \
  --dense-pt subgoals_reacher_dense.pt --floor-json Results/floor_reacher.json \
  --device cuda --out-prefix reacher_gatev4c_route \
  --json-out Results/gatev4c_route_max100.json
python -m specaccept.probes.route_horizon \
  --episodes-file reacher_gatev4c.ep150.s999.json --offsets 150 \
  --dense-pt subgoals_reacher_dense.pt --floor-json Results/floor_reacher.json \
  --device cuda --out-prefix reacher_gatev4c_route \
  --json-out Results/gatev4c_route_max150.json
echo "=== gate v4 CONFIRM routing pass complete ==="
