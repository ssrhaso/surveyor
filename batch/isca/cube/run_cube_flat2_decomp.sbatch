#!/usr/bin/env bash
#SBATCH --job-name=cube-f2d
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_f2d_%A_%a.log
# ============== CUBE DECOMPOSITION CELL: flat RH=2 @ certified budget =========
# The unified arm scored 79.88 on seeds 50-57 with ~55-58% of episodes retiring
# at replan 0-1 (c*-selected) -- i.e. much of the policy ran goal-serving at
# RH=2/600 samples. The RH=2 flat anchor (65.6-66.4) predates the 600-sample
# certification, so the seed-matched flat2@600 full-population reference is
# MISSING. This cell decomposes the unified win: if flat2@600 << 79.88, the
# c* selection + drafter mixture is doing real work; if flat2@600 ~ 79.88,
# cube's story is "RH=2 flat at proper budget was underrated" and the unified
# arm's contribution there is selection only. Either way the paper says the
# truth. Seeds 50-57 (same as k3c confirm + unified arm), certified protocol.
# ==============================================================================
set -euo pipefail
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((50 + SLURM_ARRAY_TASK_ID))
echo "=== cube flat2-decomp seed=$SEED (baseline RH=2, certified 600 samples) ==="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cube_f2d_seed${SEED}.log
