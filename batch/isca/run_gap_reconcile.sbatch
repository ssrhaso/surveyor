#!/usr/bin/env bash
#SBATCH --job-name=gap-reconcile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gap_reconcile_%j.log
# ==== M1-FM RECONCILIATION DIAGNOSTIC (descriptive; no bar, no claim) ====
# The M1-FM readout (droid2k, 400 eps) found cross-episode rel p50 = 0.496,
# refuting P-FM-1's premise (July gap_droid.json cross p50 = 0.153 on the
# 98-episode droid_lat pool). Hypothesis: population artifact — the first-
# fetched 98 episodes are scene-homogeneous, depressing cross distances.
# Test: the IDENTICAL frozen code path over the droid_lat 98-episode pool.
# Explains the discrepancy either way; alters no frozen readout.
# =========================================================================
set -euo pipefail
cd /lustre/home/ha676/le-wm/vjepa2
source /lustre/home/ha676/le-wm/.venv/bin/activate
python -m specaccept_vjepa2.calibrate_droid_pairs \
  --lat-dir /lustre/home/ha676/data/droid_lat \
  --n-eps 98 --pairs-per-ep 20 --n-cross 2000 \
  --out-json ../Results/calibration/cal_vjepa2_droid98_reconcile.json \
  --out-csv ../Results/calibration/events_vjepa2_droid98_reconcile.csv \
  2>&1 | tee ../logs/gap_reconcile_droid98.log
echo "=== DONE gap-reconcile ==="
