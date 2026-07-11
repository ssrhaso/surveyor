#!/usr/bin/env bash
# One-time env setup for a bare GPU box (no SLURM, no apptainer) -- the local
# equivalent of install_deps.sbatch. Run once from ~/le-wm:
#   bash batch/setup_local_env.sh
#
# Assumes: a working NVIDIA driver (`nvidia-smi` succeeds) and python3 with venv.
# Does NOT assume a specific CUDA version -- pip installs the CUDA-bundled
# torch wheel (cu121 by default). If `nvidia-smi` reports a driver too old for
# that wheel, override: TORCH_INDEX=https://download.pytorch.org/whl/cu118
set -eu
cd "$(dirname "$0")/.."
VENV=.venv
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}

echo "=== [0/4] GPU check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== [1/4] venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip

echo "=== [2/4] torch (CUDA wheel: $TORCH_INDEX) ==="
pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"

echo "=== [3/4] project deps (Reacher boxes: NO [env] extra -- its box2d-py dep"
echo "    cannot build under pip's isolation on these boxes, and box2d is"
echo "    PushT-only physics; Reacher needs dm_control/mujoco, added explicitly) ==="
# transformers MUST stay on the 4.x naming scheme: the LeWM encoder checkpoints
# were saved with HF ViT's old internal names (encoder.layer.N.attention.attention
# .query); transformers 5.x renamed them (layers.N.attention.q_proj) and the
# strict state_dict load fails. 4.57.1 = the exact version proven on the PushT box.
pip install --no-cache-dir 'stable-worldmodel[train]' stable-pretraining h5py hdf5plugin scikit-learn dm_control mujoco imageio imageio-ffmpeg 'transformers==4.57.1'
pip uninstall -y opencv-python || true
pip install --no-cache-dir --force-reinstall opencv-python-headless

echo "=== [4/4] sanity ==="
python -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
python -c 'import stable_worldmodel; print("stable_worldmodel OK")'
python -c 'import h5py, hdf5plugin, sklearn, stable_pretraining; print("h5py/hdf5plugin/sklearn/stable_pretraining OK")'
python -c 'import dm_control, mujoco; print("dm_control/mujoco OK (reacher env)")'
echo '=== SETUP DONE -- source .venv/bin/activate before running batch/run_reacher_local.sh ==='
