#!/usr/bin/env bash
#SBATCH --job-name=horizon-gate
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/horizon_gate_%j.log
# Episode-level horizon gate, Phase 1 (offline, pre-registered). h =
# rel(z0, z_goal) over both fixed Reacher populations x t {25..150};
# theta = disp_p50(S=10) READ from the floor probe's own JSON. Sanity
# condition (fixed in specaccept/probes/horizon_gate.py): fire >= 0.70
# at t=25, <= 0.15 at t=150, else the gate is dead. No closed loop here;
# Phase 2 runs only after this output is reviewed.
#
# Pixel-free: the raw reacher h5 was purged for the cube download
# (2026-07-13), so latents come from subgoals_reacher_dense.pt (stride-1
# dump of EVERY episode; frozen encoder in eval() => identical to pixels).
# Populations are the existing deterministic epfiles (seed 42, n=128) from
# the horizon sweeps; no h5 needed to rebuild them. No GPU needed.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
mkdir -p Results logs
python -m specaccept.probes.horizon_gate \
  --dense-pt subgoals_reacher_dense.pt --device cpu \
  --episodes-files reacher_horizon.ep.t0.json reacher_horizon150.ep.t0.json \
  --offsets 25 50 75 100 125 150 \
  --floor-json Results/floor_reacher.json --stride 10 \
  --json-out Results/horizon_gate_reacher.json \
  2>&1 | tee logs/horizon_gate_reacher.log
