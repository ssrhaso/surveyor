#!/usr/bin/env bash
#SBATCH --job-name=cube-train-gc
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_train_gc_%j.log
# GOAL-CONDITIONED cube drafter -- the arm we SHOULD have pre-registered.
#
# The mis-call: we pre-registered goal-FREE on the reasoning "cube is expert,
# goal-directed data -> the PushT regime". Wrong. The OGBench cube collector
# re-samples a NEW RANDOM TARGET within each episode, so the expert's continuation
# from any state heads toward whatever target it happened to have -- unrelated to our
# hindsight evaluation goal. That is exactly the REACHER regime (data not directed at
# our goal), where goal-free drafting is ill-posed and collapses to the floor. It did:
# 2-4% vs a 21% flat baseline (v2, corrected non-vacuous protocol).
#
# So: train goal-conditioned, identical recipe to the Reacher drafters (hindsight
# window goals, gap ~U[1,100]). Then re-run the battery with the goal-cond arm.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "=== train gdm_cube_s10_gc.pt (GOAL-CONDITIONED, subgoal-step 10, N=3) ==="
python -m specaccept.train_drafter --subgoals subgoals_cube_dense.pt \
  --out gdm_cube_s10_gc.pt --device cuda --mask 5 --subgoal-step 10 \
  --goal-cond --goal-rule window --goal-gap 1 --goal-gap-max 100 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp \
  2>&1 | tee logs/cube_train_s10_gc.log
echo "=== done: gdm_cube_s10_gc.pt ==="
