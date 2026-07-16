#!/usr/bin/env bash
#SBATCH --job-name=cube-dense
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_dense_%j.log
# Dense subgoal build (stride 1) on the frozen quentinll/lewm-cube encoder:
# encodes every frame of every episode so train_drafter --subgoal-step S carves
# any stride. Expert data -> no success filter. ~2.01M frames.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
python -m specaccept.envs.cube.build_subgoals --h5 "$H5" \
  --out subgoals_cube_dense.pt --source pretrained --encoder-id quentinll/lewm-cube \
  --device cuda --stride 1 --batch-size 256 \
  2>&1 | tee logs/cube_dense_build.log
