#!/usr/bin/env bash
#SBATCH --job-name=readout-d2k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/readout_d2k_%j.log
# ============== DROID ATTENTIVE READOUT AT ~10x DATA (pre-registered) ==============
# The 84-ep attentive lens FAILED to open droid's verification gap (job 2292631:
# width -0.082 after training). Two live explanations: (a) global readouts are
# insufficient at video scale regardless of data (gap-stat's compression story),
# (b) 84 episodes starve the learned query. The v3 drafter chain (2298734) tests
# data-scaling on the DRAFTER side; this is the matched test on the VERIFIER-LENS
# side, using the same banked 2k-episode encodes. Identical recipe to 2292631;
# only data scales (800 eps fp16-in-RAM, ~680 train / ~120 val vs 84/14).
# BARS (frozen now): OPEN (width > 0 on held-out) ==> data rescues token-space
# verification; derive tau off the new gap -> droid verification story.
# STILL CLOSED ==> "compression, not data" scope claim strengthened, consistent
# with gap-stat's prediction and complementary to v3's. Win either way.
# ===================================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
python -m specaccept.probes.learn_readout --name droid2k --attentive \
  --lat-glob "/lustre/home/ha676/data/droid2k_lat/lat_droid_ep*[0-9].npy" \
  --max-eps 800 --hop-stride 10 --batch-eps 8 \
  --out readout_droid2k.pt --device cuda \
  2>&1 | tee logs/readout_droid2k.log
echo "=== DROID-2K READOUT DONE ==="
