#!/usr/bin/env bash
#SBATCH --job-name=dwm-2room
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2room_%j.log
# ========== TWOROOM ENCODER-SWAP GAP PROBE (pre-registered, cheap kill) ==========
# Measured fact (7/22-23): LeWM's ViT-tiny saturates TwoRoom - equiv p50 0.876,
# gap inverted at EVERY scale, readout lens knife-edge => information absent
# from the encoder, park final ON THAT STACK. This probe: same pixels through
# frozen generic DINOv2 -> gap stat at hops {5,10,25}.
# PREDICTION SPACE (frozen): gap OPENS => the instrument localizes TwoRoom's
# failure to the ENCODER component (scope condition = encoder property, not
# env property); optional closed-loop follow-up (train DINO-WM on tworoom)
# only then. gap STAYS CLOSED => TwoRoom hard for generic features too;
# one-GPU-hour honest row either way. Population caveat: first 200 eps
# unfiltered (not cross-room-filtered); refine before any strong claim.
# =================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python encode_tworoom_dino.py
echo "=== gap stat (le-wm venv) ==="
cd /lustre/home/ha676/le-wm
for H in 5 10 25; do
  .venv/bin/python -m specaccept.probes.gap_stat --name tworoom_dino_h${H} \
    --lat-glob "/lustre/home/ha676/data/dinowm/tworoom_dino_lat/lat_tworoom_*.npz" \
    --equiv-stride 1 --hop-stride ${H} --tau 0.20 \
    --json-out gap_tworoom_dino_h${H}.json
done
echo "=== TWOROOM DINO GAP DONE ==="
