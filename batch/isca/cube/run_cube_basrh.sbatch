#!/usr/bin/env bash
#SBATCH --job-name=cube-basrh
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_basrhjob_%A_%a.log
# Strongest-baseline control for cube: flat goal-image planner at RH{1,2} across
# t{25,100,150}, seed 42, n=128 -- so every comparison anchors to the strongest
# flat configuration per cell (the replan-rate-is-a-brittle-tradeoff finding).
# RH5 is already in the main battery.
#   RH = {1,2}[idx//3]; OFF = {25,100,150}[idx%3]
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
RHS=(1 2); OFFSETS=(25 100 150)
RH=${RHS[$((ID / 3))]}; OFF=${OFFSETS[$((ID % 3))]}
echo "=== cube baseline RH=$RH t=$OFF seed=42 ==="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset $OFF --eval-budget $((2*OFF)) \
  --num-eval 128 --seed 42 --cem-seed 42 --horizon $RH --receding-horizon $RH \
  2>&1 | tee logs/cube_baseRH${RH}_t${OFF}_seed42.log
