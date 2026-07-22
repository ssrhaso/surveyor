#!/usr/bin/env bash
#SBATCH --job-name=vj2-stage3
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_stage3_%j.log
# ============ V-JEPA 2 TRANSPLANT: STAGE 3 (post-training probes) ===========
# (a) Drafter quality on the 8 HELD-OUT episodes (ep090-097, never trained
#     on): TokenGDM vs lerp vs no-op, pooled rel-err + movement cosine per
#     block position; the suffix-decay read at video scale.
# (b) c*-v2 instrument ladders on 3 episodes: pooled vs token-L1 metric,
#     each raw and self-floor-calibrated; can the certificate be rescued?
# Exploratory; submitted with afterok on the training job.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs

echo "=== stage 3a: drafter quality (held-out eps) ==="
python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_vj2_s10.pt \
  --episodes "/lustre/home/ha676/data/droid_lat/lat_droid_ep09*.npz" \
  --stride 10 --k 8 --device cuda \
  2>&1 | tee ../logs/vj2_probe_drafter.log

echo "=== stage 3b: c*-v2 ladders ==="
python -m specaccept_vjepa2.probe_cstar2 \
  --traj /lustre/home/ha676/data/droid_eps/droid_ep000.npz \
         /lustre/home/ha676/data/droid_eps/droid_ep007.npz \
         /lustre/home/ha676/data/droid_eps/droid_ep024.npz \
  --device cuda --cem-samples 200 --cem-steps 5 \
  2>&1 | tee ../logs/vj2_probe_cstar2.log
echo "=== STAGE 3 DONE ==="
