#!/usr/bin/env bash
#SBATCH --job-name=cube-max10
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeX10_%A_%a.log
# =============== S=10 MECHANISM SWEEP (stride frozen by decision) =============
# DECISION (user, 2026-07-15): S=10 is the universal stride across PushT/Reacher/
# Cube -- no per-env stride tuning. The S={12,20} training chain was cancelled and
# the S=15 sweep arms scancelled. gc10+gate early cells: 79.69/82.81 (>= gc15's
# 79.30 mean) -- S=10 costs nothing on cube. This array measures every cheap
# mechanism axis AT S=10 (ckpt gdm_cube_s10_gc.pt, gate on, certified protocol):
#   arm0 rh5   : H=5/RH=5 loop -- flat's strongest planner config; tests whether
#                the gate's served-goal episodes were being taxed by RH=2
#   arm1 k4    : k matched to the certified spec arm    arm2 k3: derive-k minimum
#   arm3 t15   : tau bracket below                      arm4 t30: tau bracket above
#   arm5 s4649 : champion config verbatim, seeds 46-49 (8-seed CI with arm0 of
#                2274057, which already runs gc10+gate seeds 42-45)
# EXPLORATORY -- the winning config goes to a frozen-bar confirm on fresh
# populations before anything is cited.
# ==============================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/cube logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ARMS=(rh5 k4 k3 t15 t30 s4649)
KS=(8 4 3 8 8 8)
TAUS=(0.20 0.20 0.20 0.15 0.30 0.20)
RHS=(5 2 2 2 2 2)
SEEDBASE=(42 42 42 42 42 46)
A=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((SEEDBASE[A] + SLURM_ARRAY_TASK_ID % 4))
NAME=${ARMS[$A]}; K=${KS[$A]}; TAU=${TAUS[$A]}; RH=${RHS[$A]}
echo "=== cube-max10 arm=$NAME k=$K tau=$TAU rh=$RH seed=$SEED (gc10+gate) ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate \
  --accept-tau "$TAU" --gdm-steps "$K" --horizon "$RH" --receding-horizon "$RH" \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeX10_${NAME}_seed${SEED}.log
