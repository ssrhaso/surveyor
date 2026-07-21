#!/usr/bin/env bash
#SBATCH --job-name=cube-anch4
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_anch4job_%A_%a.log
# Tighten the cube anchor: the winning config (offset=150, budget=300, RH=5) gave
# 71.1% +- 4.0 (n=128, 1 seed) vs LeWM's published 74%; a 0.7-SE difference, i.e.
# already a reproduction. Run 4 seeds (n=512) to halve the SE to ~2.0pp and settle it.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "=== cube anchor 4-seed: flat, offset=150 budget=300 RH=5, seed=$SEED (LeWM 74%) ==="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
  --horizon 5 --receding-horizon 5 \
  2>&1 | tee logs/cube_anchor4_seed${SEED}.log
