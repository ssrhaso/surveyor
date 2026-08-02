#!/usr/bin/env bash
#SBATCH --job-name=cal-vjepa2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cal_vjepa2_%j.log
# ===== M1-FM: FOUNDATION-SCALE CALIBRATION ROW (PRE-REGISTERED) =====
# docs/certification_prereg.md "M1-FM" frozen BEFORE this submission.
# Dataset-pair operating characteristic of the transplant verifier
# (pooled V-JEPA 2 ViT-g tokens, rel L2, tau=0.20) vs RECORDED DROID
# proprio ground truth over the 2,000 cached episodes. No closed loop;
# labeled dataset-pair wherever quoted. P-FM-1: degenerate accept
# (instrument's gap verdict tested at scale). P-FM-2: rho, open.
# CPU-bound numpy over mmap'd npy; gpu partition used for node-local IO.
# ====================================================================
set -euo pipefail
cd /lustre/home/ha676/le-wm/vjepa2
source /lustre/home/ha676/le-wm/.venv/bin/activate
mkdir -p ../Results/calibration ../logs
python -m specaccept_vjepa2.calibrate_droid_pairs \
  --lat-dir /lustre/home/ha676/data/droid2k_lat \
  --out-json ../Results/calibration/cal_vjepa2_droid.json \
  --out-csv ../Results/calibration/events_vjepa2_droid.csv \
  2>&1 | tee ../logs/cal_vjepa2_droid.log
echo "=== DONE cal-vjepa2 ==="
