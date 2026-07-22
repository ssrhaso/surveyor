#!/usr/bin/env bash
#SBATCH --job-name=vj2-probe
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_probe_%j.log
# ============ V-JEPA 2 TRANSPLANT: OFFLINE PROBE (stage 0) ==================
# First GPU run of the transplant package: anchor + c* ladder + floor shape +
# plumbing on the bundled Franka trajectory. Exploratory (no bar); its outputs
# decide whether tau=0.20 transfers and whether c* separates on this
# substrate. Uses the existing le-wm venv (verified 2026-07-20: torch 2.5.1
# cu121 + timm/cv2/einops/scipy all present; decord not on our import path).
# Self-contained: fetches the 11.7GB AC checkpoint into the torch-hub cache if
# missing (the pinned upstream commit points VJEPA_BASE_URL at localhost:8300,
# so the cache MUST be pre-populated; load_state_dict_from_url checks the
# cache by filename before touching the network).
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
export TORCH_HOME=/lustre/home/ha676/torch_cache
CKPT_DIR=$TORCH_HOME/hub/checkpoints
CKPT=$CKPT_DIR/vjepa2-ac-vitg.pt
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT" ]; then
  echo "=== fetching vjepa2-ac-vitg.pt (11.7GB) ==="
  wget -q --tries=3 -O "$CKPT.part" \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
  mv "$CKPT.part" "$CKPT"
fi
ls -l "$CKPT"
source "$VENV/bin/activate"
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
python -m specaccept_vjepa2.smoke_test
python -m specaccept_vjepa2.probe_offline \
  --device cuda \
  --cem-samples 200 --cem-steps 5 \
  --dump-latents lat_franka.npz \
  2>&1 | tee ../logs/vj2_probe_offline.log
