#!/usr/bin/env bash
#SBATCH --job-name=readout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/readout_%j.log
# ============ GAP-MAXIMIZING VERIFICATION READOUTS (method v-next) ============
# Derive the METRIC, not just tau: time-contrastive lens per substrate,
# trained on cached latents only (world model + planner frozen; verifier-side
# component). Read = gap before vs after on held-out episodes.
#   reacher  linear 192->64   (tail-overlap case: does the lens clean it?)
#   droid    attentive-pool over tokens, 1408->64 (pooled compression case)
#   tworoom  linear 192->64, AFTER the dense stride-1 file exists
#             (saturation case: prediction = information absent, gap stays
#              closed; a negative here SHARPENS the scope story)
# Exploratory. Closed-loop value judged separately if gaps open.
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate

echo "=== reacher (linear) ==="
python -m specaccept.probes.learn_readout --name reacher \
  --subgoals-pt subgoals_reacher_dense.pt --hop-stride 10 \
  --out readout_reacher.pt --device cuda 2>&1 | tee logs/readout_reacher.log

echo "=== droid (attentive over tokens) ==="
python -m specaccept.probes.learn_readout --name droid --attentive \
  --lat-glob "/lustre/home/ha676/data/droid_lat/lat_*.npz" --hop-stride 10 \
  --out readout_droid.pt --device cuda --batch-eps 8 \
  2>&1 | tee logs/readout_droid.log

echo "=== tworoom (linear; needs the dense file from the gapdense job) ==="
if [ -f subgoals_tworoom_s1.pt ]; then
  python -m specaccept.probes.learn_readout --name tworoom \
    --subgoals-pt subgoals_tworoom_s1.pt --hop-stride 10 \
    --out readout_tworoom.pt --device cuda 2>&1 | tee logs/readout_tworoom.log
else
  echo "subgoals_tworoom_s1.pt not present yet; run again after tworoom-gd"
fi
echo "=== READOUT LEARNING DONE ==="
