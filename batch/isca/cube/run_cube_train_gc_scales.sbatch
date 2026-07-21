#!/usr/bin/env bash
#SBATCH --job-name=cube-gc-S
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_gcS_%A_%a.log
# The cube S-curve, GOAL-CONDITIONED (S=10 is already training as 2272341).
# Justified now that the ORACLE proved cube HAS decomposition headroom (82.4% vs
# flat 70.9%, +11.5pp); so waypoints CAN win here and the scale law is worth
# measuring, as on PushT/Reacher. Goal-cond because the goal-FREE drafter fails
# (33%): the target is not inferable from the observation, so goal-free drafting is
# ill-posed on cube (the Reacher/TwoRoom regime).
# Identical recipe to the Reacher drafters: hindsight window goals, gap ~U[1,100].
#   S = {5, 15, 25}[idx]
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SS=(5 15 25)
S=${SS[$SLURM_ARRAY_TASK_ID]}
echo "=== train gdm_cube_s${S}_gc.pt (goal-cond, subgoal-step ${S}, N=3) ==="
python -m specaccept.train_drafter --subgoals subgoals_cube_dense.pt \
  --out gdm_cube_s${S}_gc.pt --device cuda --mask 5 --subgoal-step ${S} \
  --goal-cond --goal-rule window --goal-gap 1 --goal-gap-max 100 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp \
  2>&1 | tee logs/cube_train_s${S}_gc.log
