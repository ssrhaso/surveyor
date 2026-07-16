#!/usr/bin/env bash
#SBATCH --job-name=hgate-cem
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/horizon_gate_cem_%j.log
# Horizon gate v2, Phase 1 (offline, pre-registered): planner-predicted
# reachability. One flat CEM plan (RH=2, the strongbase sweep's strongest
# short-range flat config) per (episode, t); c* = rel(z_hat_H, z_goal) of
# the returned plan's predicted terminal latent; fire iff c* <= tau=0.20
# (the verifier's frozen threshold, read from the floor probe's args).
# Sanity condition baked into specaccept/probes/horizon_gate_cem.py:
# fire >= 0.70 @ t25, <= 0.15 @ t150, plus an explicit model-exploitation
# kill if fire is high at every t. Pixel-free (dense stride-1 latents);
# no env stepping, no closed loop. Phase 2 only after review.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
mkdir -p Results logs
export SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
python -m specaccept.probes.horizon_gate_cem \
  --dense-pt subgoals_reacher_dense.pt --device cuda \
  --source local --local-dir encoder_reacher \
  --episodes-files reacher_horizon.ep.t0.json reacher_horizon150.ep.t0.json \
  --offsets 25 50 75 100 125 150 \
  --floor-json Results/floor_reacher.json \
  --json-out Results/horizon_gate_cem_reacher.json \
  2>&1 | tee logs/horizon_gate_cem_reacher.log
