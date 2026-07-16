#!/usr/bin/env bash
#SBATCH --job-name=cube-dsprobe
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_dsprobe_%j.log
# OFFLINE CEILING / DETERMINISM PROBE (no CEM), on the goal-cond gc15 drafter.
# Answers the three things the closed-loop gate array cannot:
#   A1  diffusion drafter's intrinsic rel-err to the true next latent, and the
#       few-step vs 50-step spread -> the stochastic error floor.
#   A2  offline speculative-acceptance depth vs tau (demo-replay, optimistic) ->
#       cross-checks the measured closed-loop call_ratio=0.75 and picks the tau
#       for the next array. taus bracket the cube criterion floor (0.76).
#   A3  m+1 posterior multimodality. <~5% multimodal => posterior ~unimodal =>
#       a DETERMINISTIC z_cond->block regressor is well-posed and can approach
#       the oracle (the 82.4 ceiling); the win-shot is then a regressor drafter,
#       not more gating. This is the ceiling probe skipped in the gate array.
# Population is built UNBIASED (min-disp 0.0): the multi-offset non-vacuity filter
# in build_cube_horizon_episodes silently selects the steepest subset (that bug
# produced the misleading 21%), so it is deliberately NOT used here.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
EP=cube_probe.ep150.json
python batch/isca/cube/build_cube_horizon_episodes.py --h5 "$H5" --out "$EP" \
  --n 256 --offsets 150 --min-disp 0.0 --episode-min 8000 --seed 42
echo "=== dspark gates on gdm_cube_s15_gc.pt (stride 15, N=3) ==="
python -m specaccept.probes.probe_dspark_gates \
  --gdm-ckpt gdm_cube_s15_gc.pt --stride 15 \
  --h5 "$H5" --episodes-file "$EP" \
  --source pretrained --encoder-id quentinll/lewm-cube --device cuda \
  --n-probe 256 --probes a1 a2 a3 \
  --taus 0.10 0.20 0.40 0.60 0.76 0.90 99.0 \
  --json-out Results/cube_dsprobe_gc15.json
