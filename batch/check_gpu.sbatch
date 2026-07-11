#!/usr/bin/env bash
#SBATCH --job-name=check-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/u6ko/hasoshu.u6ko/containers/checkgpu_%j.log
# Two independent checks: nvidia-smi (nothing to do with torch/python) and our
# actual ffjepa package code (not just generic test imports).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
echo "=== nvidia-smi from inside the container ==="
apptainer exec --nv "$SIF" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== our actual ffjepa package: gdm_model.py + lewm_io.py ==="
apptainer exec --nv --bind /scratch:/scratch "$SIF" bash -c "
  source $VENV/bin/activate
  cd $HOME/le-wm
  python -c '
import sys; sys.path.insert(0, \".\")
from ffjepa.gdm_model import GDMConfig, GaussianDiffusion, GDMPlanner
from ffjepa import lewm_io
import torch
print(\"ffjepa.gdm_model + lewm_io import OK\")
cfg = GDMConfig(latent_dim=192, n_future=3, wg=1, hidden=512, depth=11, heads=8, mlp_ratio=4.0)
print(\"GDMConfig instantiated OK:\", cfg)
x = torch.randn(4, 192, device=\"cuda\")
print(\"real tensor on cuda:\", x.device, x.sum().item())
'
"
echo '=== DONE ==='
