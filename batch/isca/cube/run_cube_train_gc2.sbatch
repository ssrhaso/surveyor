#!/usr/bin/env bash
#SBATCH --job-name=cube-gc-S2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_gcS2_%A_%a.log
# Stride-peak refinement: the gated gc S-curve is sharp (5:67.2, 15:79.3,
# 25:68.2) with S=10 pending in the cube-max sweep. Train S={12,20} with the
# IDENTICAL recipe as run_cube_train_gc_scales.sbatch so the only variable is
# stride; a dependent battery (run_cube_gate2.sbatch, afterok) evals both at
# the champion config as soon as the checkpoints land.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SS=(12 20)
S=${SS[$SLURM_ARRAY_TASK_ID]}
echo "=== train gdm_cube_s${S}_gc.pt (goal-cond, subgoal-step ${S}, N=3) ==="
python -m specaccept.train_drafter --subgoals subgoals_cube_dense.pt \
  --out gdm_cube_s${S}_gc.pt --device cuda --mask 5 --subgoal-step ${S} \
  --goal-cond --goal-rule window --goal-gap 1 --goal-gap-max 100 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp \
  2>&1 | tee logs/cube_train_s${S}_gc.log
