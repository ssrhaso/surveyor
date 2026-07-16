#!/usr/bin/env bash
#SBATCH --job-name=cube-smoke
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_smoke_%j.log
# Cube port wiring validation (BEFORE any battery). Two stages:
#  1. --debug-keys: env instantiates on the GPU node (EGL), has set_state +
#     set_target_pos, and h5 columns/dims print (confirm prev_qpos/qvel dims,
#     block/target pos). No CEM.
#  2. baseline flat goal-image planner, n=16, short (goal_offset=25, budget 50):
#     exercises encoder(quentinll/lewm-cube) + HDF5Dataset + the set_state /
#     set_target_pos callables + info['success'] end-to-end. A sane SR (>0,
#     ideally high since the goal is inside the planner's reach at t=25 on
#     expert data) certifies the wiring; then the floor probe + battery follow.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl
echo "===== STAGE 0: ensure ogbench installed (cube_env imports ogbench.manipspace) ====="
python -c "import ogbench" 2>/dev/null && echo "ogbench present" || pip install --no-input ogbench
python -c "import ogbench; from ogbench.manipspace import lie; print('ogbench OK', ogbench.__version__ if hasattr(ogbench,\"__version__\") else \"?\")"
echo "===== STAGE 1: debug-keys (env + callable wiring) ====="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --num-eval 4 --goal-offset 25 --debug-keys \
  2>&1 | tee logs/cube_smoke_debug.log
echo "===== STAGE 2: baseline flat planner, n=16, t=25 ====="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --num-eval 16 --goal-offset 25 --eval-budget 50 --seed 42 --cem-seed 42 \
  --horizon 5 --receding-horizon 5 \
  2>&1 | tee logs/cube_smoke_baseline.log
echo "===== cube smoke done ====="
