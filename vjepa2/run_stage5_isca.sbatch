#!/usr/bin/env bash
#SBATCH --job-name=vj2-stage5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_stage5_%j.log
# ============ V-JEPA 2 TRANSPLANT: STAGE 5 (v2 drafter, extended grid) ======
# Stage-4 landed the flagship shape with the V1 drafter (far-goal t=32: spec
# 0.33 at budget 25 vs flat 0.30 at budget 400 = 15x less planning compute;
# lerp control NEGATIVE -> the win is the learned drafter, not decomposition;
# near-goal flat wins = the router's cell, c*-v2 separates exactly there).
# This rerun swaps in the RESIDUAL v2 drafter and extends offsets to
# {8,16,32,64}; the dress rehearsal for the pre-registered cut. Reading:
# does v2 widen the far-goal gap / shrink the near-goal harm, and does the
# crossover sit at t~16 where c*-v2's step is?
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
python -m specaccept_vjepa2.probe_budget_collapse \
  --ckpt gdm_vj2_s10_res.pt \
  --episodes "/lustre/home/ha676/data/droid_lat/lat_droid_ep09*.npz" \
  --device cuda --offsets 8 16 32 64 --budgets 25 100 400 \
  --anchors-per-ep 3 --max-eps 4 \
  --out budget_collapse_v2_raw.npz \
  2>&1 | tee ../logs/vj2_budget_collapse_v2.log
echo "=== STAGE 5 DONE ==="
