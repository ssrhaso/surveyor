#!/usr/bin/env bash
#SBATCH --job-name=cube-floorAB
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_floorAB_%j.log
# DERIVE-FIRST step 1 (no drafter): floor probe parts A+B on the frozen
# quentinll/lewm-cube encoder + dataset. Derives tau = the criterion floor
# (cube-position success-equivalence, benchmark's own 0.04m), and the
# displacement disp(S) sanity gate for the pre-registered S=10 (must sit well
# above the floor). Part C (sampler convergence -> k) needs the trained
# drafter and runs after training. NO closed-loop rollout here; tau is
# pre-registered from this output BEFORE the battery.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p Results
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
# CORRECTED (full-scene) criterion after the cube-only confound (see PLAN):
# match cube AND end-effector, both 0.04m. Tighter criterion -> fewer matching
# pairs, so a larger candidate pool. floor_cube.json (cube-only, confounded)
# is KEPT as the evidence for the scope-boundary finding; this writes
# floor_cube_fullscene.json.
python -m specaccept.probes.probe_floor --env cube --h5 "$H5" \
  --source pretrained --encoder-id quentinll/lewm-cube --device cuda \
  --stride 10 --cube-tol 0.04 --tau 0.20 --seed 42 \
  --n-anchors 512 --pair-pool 120000 \
  --json-out Results/floor_cube_fullscene.json \
  2>&1 | tee logs/cube_floorAB_fullscene.log
