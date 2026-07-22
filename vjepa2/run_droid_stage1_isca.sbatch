#!/usr/bin/env bash
#SBATCH --job-name=vj2-droid1
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_droid1_%j.log
# ============ V-JEPA 2 TRANSPLANT: DROID STAGE 1 ============================
# Real multi-frame episodes for the questions stage-0 could not answer:
#   (a) fetch 100 RLDS droid_100 episodes -> probe-format npz (external-left
#       cam, 5fps, base-frame poses) into ~/data/droid_eps  [own TF venv, CPU]
#   (b) probe_offline on 2 episodes with full c* ladder + floor strides
#       -> does c* separate near/far? where does the floor put tau/S?
#   (c) encode ALL episodes to token latents (~/data/droid_lat) for TokenGDM
#       training. Exploratory (no bar); tau/S derivation is pre-registered
#       AFTER reading (b), per the paper's derive-don't-tune rule.
# Idempotent: each stage skips if its output exists.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
TFVENV=/lustre/home/ha676/vj2tf
EPS=/lustre/home/ha676/data/droid_eps
LAT=/lustre/home/ha676/data/droid_lat
cd /lustre/home/ha676/le-wm/vjepa2

echo "=== stage a: fetch (TF venv) ==="
if [ ! -f "$EPS/manifest.json" ]; then
  if [ ! -f "$TFVENV/bin/activate" ]; then
    python3 -m venv "$TFVENV"
    "$TFVENV/bin/pip" -q install --upgrade pip
    "$TFVENV/bin/pip" -q install tensorflow-cpu tensorflow-datasets
  fi
  "$TFVENV/bin/python" specaccept_vjepa2/fetch_droid.py \
    --out-dir "$EPS" --num-episodes 100
else
  echo "manifest exists, skipping fetch"
fi
ls "$EPS" | head -5; ls "$EPS" | wc -l

source /lustre/home/ha676/le-wm/.venv/bin/activate
echo "=== stage b: probe on 2 real episodes ==="
for EP in droid_ep000 droid_ep001; do
  python -m specaccept_vjepa2.probe_offline \
    --traj "$EPS/$EP.npz" --device cuda \
    --cem-samples 200 --cem-steps 5 \
    --offsets 1 2 4 8 16 32 64 --strides 1 2 5 10 \
    --replan-stride 5 \
    2>&1 | tee ../logs/vj2_probe_${EP}.log
done

echo "=== stage c: encode all episodes ==="
python -m specaccept_vjepa2.encode_episodes \
  --episodes "$EPS/droid_ep*.npz" --out-dir "$LAT" --device cuda
ls "$LAT" | wc -l
echo "=== ALL STAGES DONE ==="
