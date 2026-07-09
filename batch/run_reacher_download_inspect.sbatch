#!/usr/bin/env bash
#SBATCH --job-name=reacher-download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_download_%j.log
# Overnight: download + decompress + inspect Reacher's checkpoint+dataset
# (quentinll/lewm-reacher, 23.8GB compressed), mirroring the TwoRoom
# truncation-signature analysis, to check whether Reacher ALSO has a fixable
# timeout artifact or a PushT-style genuine-difficulty failure pattern.
# CPU-only (download + tar/zstd decompress + h5py inspection, no GPU needed).
set -eu
cd /home/u6ko/hasoshu.u6ko/le-wm

bash batch/download_inspect_reacher.sh

SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
apptainer exec --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python batch/inspect_reacher.py
" 2>&1 | tee logs/inspect_reacher.log
