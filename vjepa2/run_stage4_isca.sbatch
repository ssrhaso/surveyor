#!/usr/bin/env bash
#SBATCH --job-name=vj2-stage4
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_stage4_%j.log
# ============ V-JEPA 2 TRANSPLANT: STAGE 4 (flagship probe) =================
# Budget collapse x demo agreement on the val holdout (ep090-097, disjoint
# from drafter training): flat vs spec-subgoal vs lerp-subgoal across CEM
# budgets {25,100,400} at offsets {8,32}, scored by cosine to the human
# demonstrator's true displacement + solve wall-clock. Win shape: spec >=
# flat everywhere with the gap growing at t=32, and spec at budget 25-100
# matching flat at 400 (the measured wall-clock claim at 1B scale).
# Exploratory first pass; the pre-registered version is cut AFTER this read.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
python -m specaccept_vjepa2.probe_budget_collapse \
  --ckpt gdm_vj2_s10.pt \
  --episodes "/lustre/home/ha676/data/droid_lat/lat_droid_ep09*.npz" \
  --device cuda --offsets 8 32 --budgets 25 100 400 \
  --anchors-per-ep 3 --max-eps 4 \
  --out budget_collapse_raw.npz \
  2>&1 | tee ../logs/vj2_budget_collapse.log
echo "=== STAGE 4 DONE ==="
