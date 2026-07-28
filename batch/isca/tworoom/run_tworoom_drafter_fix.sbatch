#!/usr/bin/env bash
#SBATCH --job-name=2room-draftfix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_draftfix_%j.log
# ============== TWOROOM DRAFTER REPAIR, WITH AN OFFLINE FIDELITY GATE ==============
# Both existing TwoRoom drafters emit latents about 100x too large:
#   gdm_tworoom_s10_gc.pt        drafted norm 1490 vs true 13.4  (111x)
#   gdm_tworoom_s10_gc_paired.pt drafted norm 1774 vs true 13.4  (132x)
# against a PushT control through the identical code path that is correct
# (gdm_stride10_gc.pt: drafted 13.80 vs true 14.01, ratio 0.985). So this is a
# TwoRoom-specific training failure, not a sampler or calling-convention fault.
#
# Most likely cause: TwoRoom has 11,807 training pairs, so 20 epochs is only
# 940 gradient steps for a 53M-parameter DiT. PushT trains the same recipe on
# far more data and converges. An undertrained eps-parameterised model diverges
# at sampling time.
#
# Trains four configs and gates them OFFLINE on drafted-vs-true norm ratio and
# relative error. Two axes, changed one at a time so the cause is attributable:
#   epochs 20 -> 400 (about 19k steps, comparable to PushT)
#   recipe   eps/linear -> v/cosine/min-snr 5 (the V-JEPA 2 recipe, proven on
#            384-d DINOv2 latents, which is exactly the paired half's family)
# NOTHING closed-loop runs here. No drafter may enter a battery until its norm
# ratio is near 1.0; that is an offline fidelity gate, not a result.
# ==================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate

train_one () {  # $1 subgoals  $2 out  $3 latent_dim  $4 extra recipe flags
  if [ -f "$2" ]; then echo "[skip] $2 exists"; return; fi
  echo "=== training $2 (dim $3) $4 ==="
  python -m specaccept.train_drafter \
    --subgoals "$1" --out "$2" --latent-dim "$3" --n-future 3 \
    --goal-cond --goal-rule final --mask 5 --device cuda --seed 42 \
    --epochs 400 $4 2>&1 | tail -12
}

SG192=subgoals_tworoom_s10.pt
SG576=subgoals_tworoom_s10_paired.pt
VREC="--parameterization v --schedule cosine --min-snr-gamma 5.0"

train_one "$SG192" gdm_tworoom_s10_gc_e400.pt      192 ""
train_one "$SG192" gdm_tworoom_s10_gc_e400v.pt     192 "$VREC"
train_one "$SG576" gdm_tworoom_s10_gc_paired_e400.pt  576 ""
train_one "$SG576" gdm_tworoom_s10_gc_paired_e400v.pt 576 "$VREC"

echo
echo "############ OFFLINE FIDELITY GATE ############"
for pair in \
  "$SG192:gdm_tworoom_s10_gc.pt" \
  "$SG192:gdm_tworoom_s10_gc_e400.pt" \
  "$SG192:gdm_tworoom_s10_gc_e400v.pt" \
  "$SG576:gdm_tworoom_s10_gc_paired.pt" \
  "$SG576:gdm_tworoom_s10_gc_paired_e400.pt" \
  "$SG576:gdm_tworoom_s10_gc_paired_e400v.pt" ; do
  SG=${pair%%:*}; CK=${pair##*:}
  [ -f "$CK" ] || continue
  echo
  echo "---------- $CK ----------"
  python -m specaccept.envs.tworoom.probe_paired_drafter \
    --subgoals "$SG" --ckpt "$CK" --device cuda --n 256 --steps 50 \
    2>&1 | grep -E "NORMS|drafted|m\+1|m\+2|m\+3" | head -12
done
echo
echo "=== PushT control, same code path (expected ratio about 0.985) ==="
python -m specaccept.envs.tworoom.probe_paired_drafter \
  --subgoals subgoals_dense_full.pt --ckpt gdm_stride10_gc.pt \
  --device cuda --n 256 --steps 50 2>&1 | grep -E "drafted" | head -3
echo "=== TWOROOM DRAFTER FIX DONE ==="
