#!/usr/bin/env bash
#SBATCH --job-name=cube-anchor
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_anchorjob_%A_%a.log
# BASELINE ANCHOR SWEEP. LeWM reports 74% on OGBench-Cube; our flat baseline gets
# ~21%, so OUR PROTOCOL IS WRONG -- no cube conclusion is valid until the baseline
# reproduces. Certify first, like PushT (94.6) and Reacher (84).
#
# What the data says (measured): ONE target per episode; the expert moves the cube
# from ~0.275m away to the target in ~90 steps, then IDLES ~110 steps. So:
#   * goal_offset must be large enough that start is BEFORE the cube arrives
#     (offset 200 => start = 0, goal = final frame = cube at target: the natural task)
#   * budget must exceed the ~90 steps the task physically takes (2t was starving it)
# Sweep goal_offset x budget x RH; the cell that lands ~74% is the protocol.
#   OFF = {150,200}[idx//6]; BUDGET = {200,300,400}[(idx//2)%3]; RH = {2,5}[idx%2]
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
OFFS=(150 200); BUDS=(200 300 400); RHS=(2 5)
OFF=${OFFS[$((ID / 6))]}; BUD=${BUDS[$(((ID / 2) % 3))]}; RH=${RHS[$((ID % 2))]}
echo "=== cube anchor: flat baseline  goal_offset=$OFF budget=$BUD RH=$RH  (target: LeWM 74%) ==="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset $OFF --eval-budget $BUD \
  --num-eval 128 --seed 42 --cem-seed 42 --episode-min 8000 \
  --horizon $RH --receding-horizon $RH \
  2>&1 | tee logs/cube_anchor_off${OFF}_bud${BUD}_rh${RH}.log
