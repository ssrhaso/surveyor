#!/usr/bin/env bash
#SBATCH --job-name=cube-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_train_%j.log
# Train the pre-registered S=10 GOAL-FREE drafter on Cube (expert/goal-directed
# data -> PushT regime, no --goal-cond). Identical recipe to the dev-env
# drafters. --mask 5 selects all episodes (cube in_5deg is all-ones). N=3.
# Depends on the dense build (afterok). Offline diag follows for a sanity read.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "=== train gdm_cube_s10.pt (goal-free, subgoal-step 10, N=3) ==="
python -m specaccept.train_drafter --subgoals subgoals_cube_dense.pt \
  --out gdm_cube_s10.pt --device cuda --mask 5 --subgoal-step 10 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp \
  2>&1 | tee logs/cube_train_s10.log
echo "=== DERIVE-FIRST step 2: floor probe part C (sampler convergence -> k) ==="
python -m specaccept.probes.probe_floor --env cube --h5 "$H5" \
  --source pretrained --encoder-id quentinll/lewm-cube --gdm-ckpt gdm_cube_s10.pt \
  --device cuda --stride 10 --cube-tol 0.04 --tau 0.20 --seed 42 \
  --n-anchors 512 --pair-pool 120000 \
  --ks 1 2 3 4 5 6 8 12 16 50 --json-out Results/floor_cube_full.json \
  2>&1 | tee logs/cube_floorC.log
