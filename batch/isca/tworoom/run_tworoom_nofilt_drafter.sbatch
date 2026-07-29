#!/usr/bin/env bash
#SBATCH --job-name=2room-nofilt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_nofilt_%j.log
# ========= TWOROOM: UNFILTERED-POOL PAIRED DRAFTER (item 3 of the handoff) =========
# The anchor ablation measured the success filter as a ~15pp protocol artifact,
# and the ANCHORED eval protocol (goal_proprio, unfiltered population) makes
# failed demos a well-posed training signal: conditioned on the demo's end
# frame, predict the states along the way. Dropping success+cross-room roughly
# doubles the 11,807-pair pool AND matches the serving distribution.
#
# Chain: build unfiltered paired subgoals (episodes < 4000 only, holdout kept)
#        -> train 576-d gc drafter, the PROVEN recipe (v-param/cosine/min-SNR 5,
#           400 epochs; the two ~100x-norm drafters came from 20-epoch training)
#        -> OFFLINE GATES: norm-ratio fidelity probe + best-of-k probe.
# NOTHING closed-loop runs here; the drafter may not enter a battery until the
# norm gate passes and its arm is pre-registered (docs/tworoom_paired_prereg.md).
# ===================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

SG=subgoals_tworoom_s10_paired_nofilt.pt
CK=gdm_tworoom_s10_gc_paired_nofilt_e400v.pt

if [ ! -f "$SG" ]; then
  echo "=== build unfiltered paired subgoals (episodes < 4000, stride 10) ==="
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 "$H5" --out "$SG" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --stride 10 --max-episodes 4000 --sample-mode head \
    --no-require-cross-room --no-success-filter --dino-pair
else
  echo "[skip] $SG exists"
fi

if [ ! -f "$CK" ]; then
  echo "=== train 576-d gc drafter on the unfiltered pool (proven recipe) ==="
  python -m specaccept.train_drafter \
    --subgoals "$SG" --out "$CK" --latent-dim 576 --n-future 3 \
    --goal-cond --goal-rule final --mask 5 --device cuda --seed 42 \
    --epochs 400 --parameterization v --schedule cosine --min-snr-gamma 5.0 \
    2>&1 | tail -12
else
  echo "[skip] $CK exists"
fi

echo
echo "############ OFFLINE GATE 1: norm ratio + fidelity ############"
python -m specaccept.envs.tworoom.probe_paired_drafter \
  --subgoals "$SG" --ckpt "$CK" --device cuda --n 256 --steps 50

echo
echo "############ OFFLINE GATE 2: best-of-k derivation on this drafter ############"
python -m specaccept.envs.tworoom.probe_bok \
  --subgoals "$SG" --ckpt "$CK" --device cuda --n 256 --steps 50 --seed 0
echo "=== NOFILT DRAFTER CHAIN DONE ==="
