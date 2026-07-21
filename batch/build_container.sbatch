#!/usr/bin/env bash
#SBATCH --job-name=build-pytorch-sif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ko/hasoshu.u6ko/containers/build_%j.log
# Isambard GH200 node driver is 565.57.01 (max CUDA 12.7). Use 24.12-py3
# (CUDA 12.6); newer NGC tags (e.g. 26.06-py3) need a newer driver than
# this node has and torch.cuda.is_available() silently returns False.
set -eu
mkdir -p /scratch/u6ko/hasoshu.u6ko/containers
cd /scratch/u6ko/hasoshu.u6ko/containers
echo "[node] $(hostname)"
apptainer build --force pytorch_2412.sif docker://nvcr.io/nvidia/pytorch:24.12-py3
echo "=== BUILD COMPLETE ==="
ls -la pytorch_2412.sif
