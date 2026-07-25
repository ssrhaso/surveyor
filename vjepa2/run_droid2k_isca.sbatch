#!/usr/bin/env bash
#SBATCH --job-name=vj2-d2k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_d2k_%j.log
# ============ V-JEPA 2 DATA SCALE-UP (v3): 2K DROID EPISODES ==================
# The drafter lost to no-op on 98 eps / 7k pairs; data starvation was the
# prime suspect and was never tested. This chain: mirror the first 60 shards
# of FULL droid (~55GB, ~2200 eps) -> probe-format npz -> ViT-g token latents
# as raw fp16 .npy (mmap path; ~100GB, RAM cannot hold them) -> retrain the
# V1 RECIPE (absolute grids, pooled cond, SAME arch - only data scales, so
# attribution is clean) at ~matched optimizer steps -> re-run the held-out
# drafter-quality probe. BAR (informal, frozen): v3 beats no-op at every
# position (noop/gdm > 1x); v1@98eps was 0.53-0.63x, v2-residual 0.74-0.82x.
# Idempotent per stage; mirror is DELETED after conversion (disk).
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export PYTHONUNBUFFERED=1
# Mirror runs on the LOGIN node (mirror_only.py under nohup; compute nodes
# reach GCS at ~170KB/s vs ~7MB/s from login). Gate on its marker, but only
# when the fetch still has to run: the chain deletes the mirror after a
# successful fetch, so on resubmits the manifest supersedes the marker.
test -f /lustre/home/ha676/data/droid2k_eps/manifest.json || \
test -f /lustre/home/ha676/data/droid_mirror/MIRROR_DONE || {
  echo "mirror not done; submit after mirror_only.py finishes"; exit 1; }
TFVENV=/lustre/home/ha676/vj2tf
EPS=/lustre/home/ha676/data/droid2k_eps
LAT=/lustre/home/ha676/data/droid2k_lat
MIR=/lustre/home/ha676/data/droid_mirror
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
df -h /lustre | tail -1

echo "=== stage a: fetch 2000 episodes from full droid (TF venv) ==="
if [ ! -f "$EPS/manifest.json" ]; then
  "$TFVENV/bin/python" specaccept_vjepa2/fetch_droid.py \
    --dataset droid --max-shards 60 --mirror-dir "$MIR" \
    --out-dir "$EPS" --num-episodes 2000
  rm -rf "$MIR"
  echo "[disk] mirror deleted"
else
  echo "manifest exists, skipping fetch"
fi
ls "$EPS" | wc -l; df -h /lustre | tail -1

source /lustre/home/ha676/le-wm/.venv/bin/activate
echo "=== stage b: encode all episodes to mmap-able npy latents ==="
python -m specaccept_vjepa2.encode_episodes \
  --episodes "$EPS/droid_ep*.npz" --out-dir "$LAT" --device cuda --format npy
ls "$LAT"/*.npy | wc -l; df -h /lustre | tail -1

echo "=== stage c: train v3 (v1 recipe, 20x data, matched steps) ==="
if [ ! -f gdm_vj2_s10_v3.pt ]; then
  python -m specaccept_vjepa2.train_drafter \
    --data "$LAT/lat_droid_ep*[0-9].npy" \
    --out gdm_vj2_s10_v3.pt \
    --stride 10 --n-future 3 --epochs 12 --batch 32 --val-files 100 \
    --device cuda \
    2>&1 | tee ../logs/vj2_train_v3.log
fi

echo "=== stage d: v3 drafter quality (held-out = the 100 val episodes) ==="
python - <<'EOF' > v3_holdout_glob.txt
import glob
files = sorted(glob.glob("/lustre/home/ha676/data/droid2k_lat/lat_droid_ep*[0-9].npy"))
print("\n".join(files[-100:][:16]))   # first 16 of the holdout for the probe
EOF
python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_vj2_s10_v3.pt \
  --episodes $(cat v3_holdout_glob.txt | tr '\n' ' ') \
  --stride 10 --k 8 --device cuda \
  2>&1 | tee ../logs/vj2_probe_drafter_v3.log
echo "=== D2K CHAIN DONE ==="
