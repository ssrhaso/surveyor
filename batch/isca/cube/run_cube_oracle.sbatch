#!/usr/bin/env bash
#SBATCH --job-name=cube-oracle
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_oraclejob_%A_%a.log
# THE DECISIVE CONTROL for the cube negative: ORACLE waypoints (ground-truth demo
# latents at start, start+S, ..., goal). This bounds EVERY waypoint method.
#   - If the oracle ALSO loses to flat planning, then no drafter can win on cube:
#     the task has no decomposition headroom (goal already inside the planner's
#     reach, flat = 75.8%) and the negative is STRUCTURAL, not a deficiency of our
#     drafter. This is exactly the Reacher-native result (oracle 84.2 == flat 84.0).
#   - If the oracle BEATS flat, then headroom exists and our drafter is simply weak
#     -- a very different (and much worse for us) conclusion.
# Certified protocol: offset=150, budget=300, 600 CEM samples; oracle stride S=10,
# matched loop RH=2. 4 seeds, n=512.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "=== cube ORACLE waypoints (S=10): seed=$SEED  [flat=75.8, drafter~29] ==="
python -m specaccept.envs.cube.eval --subgoal oracle --stride 10 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/cubeF_oracle_seed${SEED}.log
