#!/usr/bin/env bash
#SBATCH --job-name=install-deps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ko/hasoshu.u6ko/containers/install_%j.log
# Full working recipe, consolidated from three failed attempts:
#   1. stable-worldmodel[env] needs box2d-py, which needs swig to build (missing
#      in the container) -- install swig via pip first.
#   2. This node is headless: opencv-python (GUI variant) needs libxcb/libGL
#      that aren't present -- force opencv-python-headless instead.
#   3. pip install silently replaces the container's own driver-matched torch
#      with a generic newer-CUDA wheel while resolving stable-worldmodel's
#      deps. A constraints-file pin does NOT fix this (pip freeze can't see
#      the container's NVIDIA-custom torch build before anything else has
#      touched the venv). The actual fix: the replacement lands as a
#      venv-LOCAL shadow install (the container's own /usr/local torch is
#      read-only inside the SIF, so pip can't touch it) -- uninstalling the
#      shadowed torch/torchvision/torchmetrics lets the container's own
#      build show back through via --system-site-packages.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
apptainer exec --nv --bind /scratch:/scratch "$SIF" python -m venv --system-site-packages "$VENV"
apptainer exec --nv --bind /scratch:/scratch "$SIF" bash -c "
  source $VENV/bin/activate
  pip install --no-cache-dir swig
  pip install --no-cache-dir 'stable-worldmodel[train,env]' stable-pretraining h5py scikit-learn
  pip uninstall -y opencv-python
  pip install --no-cache-dir --force-reinstall opencv-python-headless
  pip uninstall -y torch torchvision torchmetrics
  echo '=== INSTALLED ==='
  python -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
  python -c 'import stable_worldmodel; from stable_worldmodel.envs.pusht.env import PushT; print(\"stable_worldmodel + PushT OK\")'
  python -c 'import h5py, sklearn, stable_pretraining; print(\"h5py/sklearn/stable_pretraining OK\")'
"
echo '=== JOB DONE ==='
