#!/usr/bin/env bash
#SBATCH --job-name=vj2-ra32v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_ra32v3_%A_%a.log
# ============ ROLLOUT AGREEMENT, t=32, v3 DRAFTER (20x-data arm) ============
# Same 12 eps / anchors / budgets as the v1 array; --arms spec only (flat and
# lerp rows are seed-identical to v1's - no wasted GPU). Depends afterok on the
# d2k chain (2298760), so gdm_vj2_s10_v3.pt exists. These eps are the v3
# TRAINING HOLDOUT - contamination-clean by construction. READ (frozen):
# spec-v3 vs spec-v1 on paired anchors = does 20x data change PLANNING value
# (not just latent fidelity)? Joint with the gap-stat prediction: if v3 fails
# stage-d's no-op bar AND spec-v3 == spec-v1 here, data-scaling is dead on the
# pooled recipe and the instrument called it prospectively.
# ============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1 LC_ALL=C
cd /lustre/home/ha676/le-wm/vjepa2
source ../.venv/bin/activate
test -f gdm_vj2_s10_v3.pt || { echo "v3 ckpt missing"; exit 1; }
LAT=/lustre/home/ha676/data/droid2k_lat
mapfile -t HOLD < <(ls $LAT/lat_droid_ep*[0-9].npy | sort | tail -100 | sed -n '17,28p')
EP=${HOLD[$SLURM_ARRAY_TASK_ID]}
BN=$(basename "$EP" .npy)
echo "=== ra32 v3 ep=$BN ==="
for B in 25 100 400; do
  python -m specaccept_vjepa2.probe_rollout_agreement \
    --ckpt gdm_vj2_s10_v3.pt --episodes "$EP" \
    --goal-offset 32 --budget $B --anchors-per-ep 3 --max-eps 1 \
    --arms spec \
    --out ra_v3_t32_b${B}_${BN}.npz --device cuda
done
echo "=== RA32 v3 task $SLURM_ARRAY_TASK_ID DONE ==="
