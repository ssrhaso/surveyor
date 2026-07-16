#!/usr/bin/env bash
#SBATCH --job-name=gate-validity
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gate_validity_%j.log
# Diagnostic post-mortem (NOT pre-registered): score gate signals (h, c*,
# drafter-arrival) per episode against the recorded flat/spec outcomes in
# runs/reacher/traces_*.pt. An episode-level gate's closed-loop SR is exactly
# determined by recorded outcomes, so gates are evaluated with zero new
# rollouts. Also checks the detour mechanism (first drafts moving away from
# the goal) at every t. Pixel-free; no env; no closed loop.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
mkdir -p Results logs
export SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
python -m specaccept.probes.gate_validity \
  --dense-pt subgoals_reacher_dense.pt --device cuda \
  --source local --local-dir encoder_reacher \
  --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 \
  --floor-json Results/floor_reacher.json \
  --json-out Results/gate_validity_reacher.json \
  2>&1 | tee logs/gate_validity_reacher.log
