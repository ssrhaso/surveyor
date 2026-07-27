#!/usr/bin/env bash
#SBATCH --job-name=2room-pairlens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_pairlens_%j.log
# ============ TWOROOM PAIRED VERIFIER: THE PRESCRIBED LENS ==================
# The raw pooled-DINOv2 gap on TwoRoom is CLOSED at every hop:
#   equiv(criterion, 16px) p90 = 0.2097   vs   hop(10) p10 = 0.0736
# but the BULKS are separated ~1.7x (equiv p50 0.098, hop10 p50 0.165, cross
# p50 0.226). That is the reacher/wall shape, and the instrument's decision
# rule for it is "bulks separated but tails overlap -> lens or router", not a
# fixed threshold. So this job trains the time-contrastive lens in exactly the
# verifier's space and reports whether the gap opens.
#
# This is the instrument PRESCRIBING the mechanism before the closed loop runs,
# which is the same move that was frozen for wall (P4) and tworoom-on-DINO.
# If the lens does not open the gap, that is the reported result and the fixed
# tau arm stands alone -- no re-tuning against a closed-loop number.
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
LATDIR=/lustre/home/ha676/data/tworoom/paired_lat

echo "=== STEP 1: dump per-episode DINOv2 latents in the verifier's space ==="
if [ ! -f "$LATDIR/lat_tworoom_paired_0000.npz" ]; then
  python -m specaccept.envs.tworoom.probe_dino_gap \
    --h5 "$H5" --device cuda --hops 5 10 25 \
    --episodes 600 --episode-max 4000 \
    --dump-latents "$LATDIR" \
    --out Results/gap_stat/gap_tworoom_paired_600.json \
    2>&1 | tee logs/2room_paired_gap600.log
fi

echo "=== STEP 2: lens at the serving hop (S=10) ==="
python -m specaccept.probes.learn_readout --name tworoom_paired_h10 \
  --lat-glob "$LATDIR/lat_tworoom_paired_*.npz" \
  --hop-stride 10 --out readout_tworoom_paired_h10.pt --device cuda \
  2>&1 | tee logs/2room_paired_lens_h10.log

echo "=== TWOROOM PAIRED LENS DONE ==="
echo "Read gap_before/gap_after/tau_derived out of the log above, record them in"
echo "docs/tworoom_paired_prereg.md, THEN submit the battery."
