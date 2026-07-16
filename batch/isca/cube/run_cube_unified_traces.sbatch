#!/usr/bin/env bash
#SBATCH --job-name=unif-cube-tr
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/unif_cube_tr_%A_%a.log
# Trace-dumping rerun of the cube unified arm (identical config to
# run_cube_unified.sbatch, seeds 50-57) so the paper's per-branch
# decomposition TBD can be computed: per-episode success x c*_first x
# retire_replan -> success of retired-at-first-replan episodes vs drafting
# episodes. SR expected to reproduce 79.9 within seed noise (same seeds,
# same protocol; the dump changes nothing in the loop).
set -euo pipefail
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs runs/cube
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((50 + SLURM_ARRAY_TASK_ID))
echo "=== unif-cube-traces seed=$SEED ==="
python -m specaccept.envs.cube.eval --subgoal unified \
  --gdm-ckpt gdm_cube_s10_gc.pt --gdm-steps 3 --accept-tau 0.20 \
  --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  --dump-traces runs/cube/traces_unified_seed${SEED}.pt \
  2>&1 | tee logs/unif_cube_tr_seed${SEED}.log
