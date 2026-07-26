#!/usr/bin/env bash
# Minimal venv for the DINO-WM PushT planning path (no MuJoCo/TF/JAX/robosuite).
set -e
cd /lustre/home/ha676/dino_wm
PY=python3.11
$PY -m venv .venv
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install numpy==1.26.4
pip -q install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip -q install hydra-core==1.2.0 hydra-submitit-launcher submitit einops \
  "gym==0.23.1" pymunk==6.8.0 pygame shapely opencv-python scikit-image \
  matplotlib imageio imageio-ffmpeg decord wandb scipy pillow
python - <<'EOF'
import torch, hydra, gym, pymunk, pygame, cv2, skimage, decord, einops, wandb
print("IMPORTS OK torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo VENV_DONE
