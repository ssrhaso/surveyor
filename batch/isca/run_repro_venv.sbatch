#!/usr/bin/env bash
#SBATCH --job-name=repro-venv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/repro_venv_%j.log
# ==== REPRO SMOKE stage 1: fresh venv from the bootstrap pip spec ====
# docs/repro_smoke_prereg.md (registered first). Production .venv untouched.
set -euo pipefail
module load Anaconda3/2023.07-2
cd /lustre/home/ha676/le-wm
rm -rf .venv_repro
python3 -m venv .venv_repro
source .venv_repro/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir 'stable-worldmodel[train]' stable-pretraining h5py hdf5plugin scikit-learn dm_control mujoco imageio imageio-ffmpeg 'transformers==4.57.1'
pip uninstall -y opencv-python || true
pip install --no-cache-dir --force-reinstall opencv-python-headless
python -c 'import torch; print("torch", torch.__version__)'
python -c 'import stable_worldmodel; print("stable_worldmodel OK")'
python -c 'import h5py, hdf5plugin, sklearn, stable_pretraining, dm_control; print("deps OK")'
echo REPRO-VENV-DONE
