#!/usr/bin/env bash
#SBATCH --job-name=vj2-ra16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_ra16_%A_%a.log
# ============== ROLLOUT AGREEMENT, t=16 (the crossover cell), v1 ==============
# Stage 4 put flat's win at t=8 and spec's at t=32; c*-v2's calibrated step sat
# at t~8-16. This slice measures the crossover region on the same 12 eps as the
# t=32 array (paired across t), budgets {25,400}, arms flat/spec only.
# DESCRIPTIVE (no bars): maps where the router's cell boundary falls at power.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1 LC_ALL=C
cd /lustre/home/ha676/le-wm/vjepa2
source ../.venv/bin/activate
LAT=/lustre/home/ha676/data/droid2k_lat
mapfile -t HOLD < <(ls $LAT/lat_droid_ep*[0-9].npy | sort | tail -100 | sed -n '17,28p')
E1=${HOLD[$((2 * SLURM_ARRAY_TASK_ID))]}
E2=${HOLD[$((2 * SLURM_ARRAY_TASK_ID + 1))]}
echo "=== ra16 v1 eps=$(basename "$E1") $(basename "$E2") ==="
for B in 25 400; do
  python -m specaccept_vjepa2.probe_rollout_agreement \
    --ckpt gdm_vj2_s10.pt --episodes "$E1" "$E2" \
    --goal-offset 16 --budget $B --anchors-per-ep 3 --max-eps 2 \
    --arms flat,spec \
    --out ra_v1_t16_b${B}_task${SLURM_ARRAY_TASK_ID}.npz --device cuda
done
echo "=== RA16 v1 task $SLURM_ARRAY_TASK_ID DONE ==="
