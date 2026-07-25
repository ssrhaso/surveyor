#!/usr/bin/env bash
#SBATCH --job-name=vj2-ra32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_ra32_%A_%a.log
# ========== HIGH-POWER BUDGET COLLAPSE, t=32 (the 15x flagship decider) ==========
# The 7/22 pre-registered cut FAILED on the 2-step displacement-cosine proxy,
# which is anchor-noise-dominated (spec swung 0.33->0.02 on 3-vs-5 anchors).
# This is the powered re-test on the PROPER instrument: full receding-horizon
# MPC inside the AC model, K=20 executed steps/anchor (~10x signal), whole-pose
# trajectory scoring. 12 held-out droid2k eps (v3 holdout slice [17..28], not
# stage-d's 16) x 3 anchors = n=36 per cell; budgets {25,100,400}; arms
# flat/spec(v1)/lerp/route. NOTE (stated, not hidden): pooled verifier is
# degenerate-accept on DROID (gap-stat) -> spec == sequential block consumption.
# v1 drafter trained on droid_100 (98 eps); overlap with these full-droid eps
# possible but <=5% - the v3 arm (afterok array) is contamination-clean.
# PRE-REGISTERED BARS (frozen; disp_cos PRIMARY, final_err/rmse secondary):
#   B1 collapse: mean disp_cos spec@25 >= flat@400 - 0.02
#   B2 low-budget: mean disp_cos spec@25 > flat@25
#   B3 control:   lerp@B <= flat@B at every budget
#   B4 router:    route@B >= max(flat,spec)@B - 0.02 at every budget
# B1&B2 pass -> 15x claim RESURRECTED with power (s/anchor col = wall-clock fig);
# B1&B2 fail -> claim BURIED; section stays mechanism-transfer + honest negative.
# =================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1 LC_ALL=C
cd /lustre/home/ha676/le-wm/vjepa2
source ../.venv/bin/activate
LAT=/lustre/home/ha676/data/droid2k_lat
mapfile -t HOLD < <(ls $LAT/lat_droid_ep*[0-9].npy | sort | tail -100 | sed -n '17,28p')
EP=${HOLD[$SLURM_ARRAY_TASK_ID]}
BN=$(basename "$EP" .npy)
echo "=== ra32 v1 ep=$BN ==="
for B in 25 100 400; do
  python -m specaccept_vjepa2.probe_rollout_agreement \
    --ckpt gdm_vj2_s10.pt --episodes "$EP" \
    --goal-offset 32 --budget $B --anchors-per-ep 3 --max-eps 1 \
    --arms flat,spec,lerp,route \
    --out ra_v1_t32_b${B}_${BN}.npz --device cuda
done
echo "=== RA32 v1 task $SLURM_ARRAY_TASK_ID DONE ==="
