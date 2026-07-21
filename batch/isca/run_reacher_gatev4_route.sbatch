#!/usr/bin/env bash
#SBATCH --job-name=gatev4-route
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev4_route_%j.log
# ==================== GATE V4 TERNARY ROUTER: ROUTING PASS ==================
# Stage 1 of 2 (eval array runs afterok). Builds FRESH populations (builder
# seed 888; 42 was analysis, 777 was the v3 confirm) and routes every
# (episode, t) via specaccept/probes/route_horizon.py:
#   c*_2 <= tau=0.20 -> flat2 ; elif c*_5 <= tau -> flat5 ; else spec
# c* = planner-predicted terminal discrepancy of one flat CEM plan from the
# episode's true start latent (gate v3's exact signal; RH=5 adds the deep
# branch v3 structurally lacked; the diagnosed t50 miss). Routing CEM seed
# 42, one decision per (episode, t), episode-start information only.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /lustre/home/ha676/le-wm
mkdir -p logs Results
source "$VENV/bin/activate"
export MUJOCO_GL=egl
python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
  --out reacher_gatev4.ep100.s888.json --n 128 --max-offset 100 \
  --episode-min 8000 --seed 888
python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
  --out reacher_gatev4.ep150.s888.json --n 128 --max-offset 150 \
  --episode-min 8000 --seed 888
python -m specaccept.probes.route_horizon \
  --episodes-file reacher_gatev4.ep100.s888.json --offsets 25 50 100 \
  --dense-pt subgoals_reacher_dense.pt --floor-json Results/floor_reacher.json \
  --device cuda --out-prefix reacher_gatev4_route \
  --json-out Results/gatev4_route_max100.json
python -m specaccept.probes.route_horizon \
  --episodes-file reacher_gatev4.ep150.s888.json --offsets 150 \
  --dense-pt subgoals_reacher_dense.pt --floor-json Results/floor_reacher.json \
  --device cuda --out-prefix reacher_gatev4_route \
  --json-out Results/gatev4_route_max150.json
echo "=== gate v4 routing pass complete ==="
