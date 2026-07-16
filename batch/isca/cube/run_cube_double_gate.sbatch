#!/usr/bin/env bash
#SBATCH --job-name=cube-dbl-gate
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_dbl_gate_%j.log
# FEASIBILITY GATE for the genuine long-horizon manipulation test (cube-DOUBLE).
#
# Why: cube-SINGLE is structurally one pick-and-place -- the cube's start->goal
# displacement saturates (~0.23m at EVERY offset), so the goal never outruns the
# planner's lookahead and drafting cannot help by construction. Multi-cube is a
# SEQUENCE of pick-and-places = genuinely long horizon, which is where the regime
# map says drafting should win. This gate answers, cheaply, whether it is even
# possible with the assets we have:
#   (1) does ogbench's cube-double dataset carry renderable state (qpos/qvel)?
#   (2) does the LeWM single-cube WORLD MODEL (encoder + latent dynamics that CEM
#       plans through) transfer to a 2-cube scene? If the dynamics cannot represent
#       two cubes, planning fails and multi-cube needs a NEW world model (big lift).
# Decision rule: if the dataset is renderable AND the flat baseline is well above
# floor on rendered cube-double, multi-cube is GO. Otherwise cube-single's
# (predicted) negative stands and we do not sink days.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "===== STAGE 1: is cube-double renderable? ====="
python /tmp/dbl.py 2>&1 | grep -viE "warning|noqa"
echo "===== gate stage 1 done ====="
