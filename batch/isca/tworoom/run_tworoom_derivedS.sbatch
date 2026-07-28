#!/usr/bin/env bash
#SBATCH --job-name=2room-derivedS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_derivedS_%j.log
# ============ TWOROOM AT THE DERIVED STRIDE (tests the new rule) ============
# probe_stride_saturation measured hop(S) p50 / cross p50 in LeWM's TwoRoom
# latent space:
#   S=1 0.626 | S=2 0.756 | S=3 0.868 | S=5 0.942 | S=10 0.988
# versus the working substrates at their tuned strides (PushT S=10 0.526,
# Reacher S=10 0.639, Cube S=15 0.624). So S=10, the stride every TwoRoom run
# to date has used, serves a subgoal 98.8 percent of the way to an unrelated
# latent, and the LARGEST stride inside the informative range is S=2.
#
# This job builds and runs TwoRoom at that derived stride, end to end, with the
# repaired training recipe. It is the prescriptive test of the saturation rule:
# the rule already explains the failures diagnostically, and this asks whether
# it also fixes them.
#
# FROZEN PREDICTION S2 (before this ran): S=2 still fails to beat flat. The
# subgoal is now inside the informative range, but 2 frames is about 10 px of
# motion against a task needing roughly 190 px, so it is too close to decompose
# anything. If it DOES beat flat, the saturation rule is prescriptive and not
# merely diagnostic, which is the stronger result.
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

SG=subgoals_tworoom_s2_paired.pt
CKPT=gdm_tworoom_s2_gc_paired.pt

echo "=== STEP 1: paired subgoals at the derived stride S=2 ==="
if [ ! -f "$SG" ]; then
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 "$H5" --out "$SG" --source pretrained \
    --encoder-id quentinll/lewm-tworooms --device cuda \
    --stride 2 --max-episodes 4000 --sample-mode head \
    --require-cross-room --dino-pair 2>&1 | tail -8
fi

echo "=== STEP 2: drafter, repaired recipe (v / cosine / min-SNR 5) ==="
if [ ! -f "$CKPT" ]; then
  python -m specaccept.train_drafter \
    --subgoals "$SG" --out "$CKPT" --latent-dim 576 --n-future 3 \
    --goal-cond --goal-rule final --mask 5 --device cuda --seed 42 \
    --epochs 400 --parameterization v --schedule cosine --min-snr-gamma 5.0 \
    2>&1 | tail -6
fi

echo "=== STEP 3: OFFLINE FIDELITY GATE (no closed-loop cell before this) ==="
python -m specaccept.envs.tworoom.probe_paired_drafter \
  --subgoals "$SG" --ckpt "$CKPT" --device cuda --n 256 --steps 50 \
  2>&1 | grep -E "NORMS|ratio|m\+1"
RATIO=$(python - "$SG" "$CKPT" <<'PY'
import sys, torch
from specaccept.drafter import load_gdm_planner
from specaccept.paired import LEWM_DIM
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
lat = blob["latents"].float()
pl = load_gdm_planner(sys.argv[2], device="cuda")
g = torch.Generator(device="cuda").manual_seed(0)
idx = torch.randperm(lat.shape[0])[:64]
z = lat[idx].cuda()
with torch.no_grad():
    blk = pl.sample_sequence(z, n_steps=50, generator=g,
                             z_goal_native=z if pl.goal_cond else None)
r = (blk[:, 0, :LEWM_DIM].norm(dim=-1).mean()
     / lat[:, :LEWM_DIM].norm(dim=-1).mean().cuda()).item()
print(f"{r:.4f}")
PY
)
echo "[gate] drafted/true LeWM norm ratio = $RATIO"
python -c "import sys; r=float('$RATIO'); sys.exit(0 if 0.5 < r < 2.0 else 1)" || {
  echo "GATE FAILED: drafter norm ratio $RATIO outside [0.5, 2.0]; no battery run."
  exit 1; }

echo "=== STEP 4: battery at S=2, flat / gdm / specpaired ==="
for OFF in 25 75; do
  BUD=$((2 * OFF))
  MODE=$([ "$OFF" -le 25 ] && echo short || echo long)
  for SEED in 42 43; do
    for ARM in baseline gdm specpaired; do
      EXTRA=()
      [ "$ARM" != "baseline" ] && EXTRA=(--gdm-ckpt "$CKPT" --gdm-steps 50 --accept-tau 0.20)
      LOG=logs/2room_S2_${ARM}_t${OFF}_seed${SEED}_${SLURM_JOB_ID}.log
      echo "--- arm=$ARM t=$OFF seed=$SEED"
      python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
        --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
        --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
        --start final --eval-filter success --require-cross-room --episode-min 4000 \
        --num-eval 64 --seed "$SEED" --cem-seed "$SEED" \
        "${EXTRA[@]}" 2>&1 | tee "$LOG" | grep -E "SR =|call_ratio"
    done
  done
done
echo "=== TWOROOM DERIVED-S DONE ==="
