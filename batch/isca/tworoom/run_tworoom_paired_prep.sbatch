#!/usr/bin/env bash
#SBATCH --job-name=2room-pairprep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_pairprep_%j.log
# ======= TWOROOM PAIRED-VERIFIER PREP: gap probe -> subgoals -> drafter =======
# Why this exists. LeWM's TwoRoom encoder cannot verify: consecutive frames sit
# at rel L2 p50 0.876 against a random-pair baseline of 1.456, and equiv p90
# (1.393) is ABOVE hop p10 (1.018), so no tau separates arrival from
# non-arrival (Results/gap_stat/gap_tworoom*.json). Flat CEM planning is
# unaffected -- it never reads that metric -- and already anchors at 68 percent
# (t=25) collapsing to 31 (t=75), which is exactly the regime the method
# targets. So we keep LeWM's world model and planner and move ONLY the
# verifier into frozen-DINOv2 space (specaccept/paired.py).
#
# NO CLOSED-LOOP CELL RUNS HERE. This job produces the instrument reading and
# the drafter; the bars get pre-registered from the measured tau BEFORE the
# battery is submitted.
# =============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs Results/gap_stat
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
echo "H5=$H5"

echo "=== STEP 1: verification gap in the DINOv2 verifier space (derives tau) ==="
python -m specaccept.envs.tworoom.probe_dino_gap \
  --h5 "$H5" --device cuda --hops 5 10 25 \
  --episodes 250 --episode-max 4000 \
  --out Results/gap_stat/gap_tworoom_paired.json \
  2>&1 | tee logs/2room_paired_gap.log

echo "=== STEP 2: paired subgoals S=10 (train episodes < 4000, cross-room) ==="
SG=subgoals_tworoom_s10_paired.pt
if [ ! -f "$SG" ]; then
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 "$H5" --out "$SG" --source pretrained \
    --encoder-id quentinll/lewm-tworooms --device cuda \
    --stride 10 --max-episodes 4000 --sample-mode head \
    --require-cross-room --dino-pair \
    2>&1 | tee logs/2room_paired_build.log
fi

echo "=== STEP 3: joint 576-d drafter (one sample emits both halves) ==="
CKPT=gdm_tworoom_s10_gc_paired.pt
if [ ! -f "$CKPT" ]; then
  python -m specaccept.train_drafter \
    --subgoals "$SG" --out "$CKPT" --latent-dim 576 --n-future 3 \
    --goal-cond --goal-rule final --mask 5 --device cuda --seed 42 \
    2>&1 | tee logs/2room_paired_train.log
fi

echo "=== TWOROOM PAIRED PREP DONE ==="
echo "NEXT: read tau from Results/gap_stat/gap_tworoom_paired.json (hop S=10),"
echo "      write the frozen bars into docs/tworoom_paired_prereg.md,"
echo "      THEN submit run_tworoom_paired_battery.sbatch."
