#!/usr/bin/env bash
#SBATCH --job-name=vj2-cut
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_cut_%j.log
# ============ V-JEPA 2 FLAGSHIP: PRE-REGISTERED CUT =========================
# The citable version of the budget-collapse x demo-agreement result.
# Drafter = V1 (gdm_vj2_s10.pt; v2 residual REGRESSED planning value and is
# retired to the instrumentation-lesson section). Pool = ALL 8 held-out
# episodes ep090-097 (stages 4/5 accidentally used only the 4 shortest;
# ep094-097 reach T=113-374 so t=64 exists), 5 anchors/ep, offsets
# {8,16,32,64}, budgets {25,100,400}. ROUTER-COMPOSITE arm enabled:
# per-anchor token-calibrated certificate, theta=0.07 (frozen from the
# stage-3 ladder step at +0.06..+0.08), routes flat vs spec.
# BARS (FROZEN NOW, before any cell; pooled over all anchors):
#   B1 capability:  spec >= flat at t=32 at EVERY budget.
#   B2 efficiency:  spec@25 >= flat@400 - 0.03 at t=32.
#   B3 control:     lerp <= flat at t=32 (else the drafter attribution falls).
#   REPORTED (not gated): t=8/16 (flat-favored, the router's cells), t=64,
#   route vs max(flat, spec) per cell.
# Report as-is, no amendment on failure.
# Usage: sbatch run_flagship_cut_isca.sbatch '<ep glob>' <tag>
# Split into two parallel halves (per-episode checkpoint + resume in the
# probe, so a timeout costs at most one episode):
#   sbatch ... 'lat_droid_ep09[0-3].npz' cutA
#   sbatch ... 'lat_droid_ep09[4-7].npz' cutB
# Merge for the final table: rerun with the merged glob and both npz rows
# concatenated (or read both npz in the aggregation notebook).
# ==============================================================================
set -euo pipefail
GLOB=${1:?usage: sbatch run_flagship_cut_isca.sbatch '<ep glob>' <tag>}
TAG=${2:?tag required}
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
python -m specaccept_vjepa2.probe_budget_collapse \
  --ckpt gdm_vj2_s10.pt \
  --episodes "/lustre/home/ha676/data/droid_lat/$GLOB" \
  --device cuda --offsets 8 16 32 64 --budgets 25 100 400 \
  --anchors-per-ep 5 --max-eps 8 \
  --router-theta 0.07 --cstar-samples 100 \
  --out budget_collapse_${TAG}_raw.npz \
  2>&1 | tee ../logs/vj2_flagship_${TAG}.log
echo "=== FLAGSHIP ${TAG} DONE ==="
