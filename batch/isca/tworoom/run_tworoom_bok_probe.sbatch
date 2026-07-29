#!/usr/bin/env bash
#SBATCH --job-name=2room-bokprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_bokprobe_%j.log
# Offline k derivation for best-of-k paired drafting. Derivation rule is fixed
# in the probe's docstring (smallest k with >= 80% of the k=16 median-score
# gain); kill criteria: no gain to select on, or selection degrades the LeWM
# half. Runs on the 400-epoch v-param drafter that passed the norm-ratio gate.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
python -m specaccept.envs.tworoom.probe_bok \
  --subgoals subgoals_tworoom_s10_paired.pt \
  --ckpt gdm_tworoom_s10_gc_paired_e400v.pt \
  --device cuda --n 256 --steps 50 --seed 0
