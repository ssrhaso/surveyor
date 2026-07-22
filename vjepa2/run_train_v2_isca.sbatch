#!/usr/bin/env bash
#SBATCH --job-name=vj2-train2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_train2_%j.log
# ============ V-JEPA 2 TRANSPLANT: RESIDUAL DRAFTER (v2) + RE-PROBE =========
# v1 verdict (stage 3): absolute-grid drafting from a pooled summary LOSES to
# no-op 2x (rel 0.26 vs 0.14); the drafter must regenerate 256 tokens of
# static scene from a 1408-d thumbnail while no-op keeps the grid for free.
# v2 fix: denoise RESIDUALS (block - cond grid); static content is inherited,
# capacity goes to what changes. Same data, stride, arch; residual stats for
# the denoised quantity, frame stats for conditioning. Trains then re-runs
# the held-out drafter-quality probe in the same job. BAR (informal, frozen
# here): v2 must beat no-op at every position (noop/gdm > 1x) to be usable.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs

python -m specaccept_vjepa2.train_drafter \
  --data "/lustre/home/ha676/data/droid_lat/lat_*.npz" \
  --out gdm_vj2_s10_res.pt --residual \
  --stride 10 --n-future 3 --epochs 150 --batch 32 --device cuda \
  2>&1 | tee ../logs/vj2_train_drafter_v2.log

echo "=== v2 drafter quality (held-out eps) ==="
python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_vj2_s10_res.pt \
  --episodes "/lustre/home/ha676/data/droid_lat/lat_droid_ep09*.npz" \
  --stride 10 --k 8 --device cuda \
  2>&1 | tee ../logs/vj2_probe_drafter_v2.log
echo "=== V2 CHAIN DONE ==="
