#!/usr/bin/env bash
#SBATCH --job-name=2room-pairbat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_pairbat_%A_%a.log
# ========= TWOROOM PAIRED-VERIFIER BATTERY (pre-registered, frozen bars) =========
# Bars and every measured input frozen in docs/tworoom_paired_prereg.md BEFORE
# this ran. Nothing here is swept.
#
# Flat anchor at this exact protocol (raw logs, n=64, seeds 42/43):
#     t=25  65.62 / 62.50  -> mean 64.06
#     t=75  34.38 / 20.31  -> mean 27.34
# The 36.7pp collapse with horizon is the headroom the method targets.
#
# Measured instrument (pooled DINOv2, serving space, 600 train episodes):
#   equiv(16px) p50 0.098 / p90 0.210 | hop10 p10 0.073 / p50 0.163 | cross p50 0.226
#   -> encoder swap RESTORED metric structure (LeWM: equiv 0.876 vs cross 1.456)
#   -> but gap [0.210, 0.073] CLOSED at every hop; lens narrowed it (-0.080 ->
#      -0.027) and never opened it.
#   -> criterion-floor rule (in scope, ratio 1.16 << 3x guard) gives the band
#      [0.098, 0.273]; the frozen global tau=0.20 lies inside it.
# The gap statistic predicts the fixed-tau arms UNDERPERFORM; the criterion-floor
# rule says tau=0.20 is usable. This battery adjudicates between our own rules.
#
# SIX ARMS x t in {25,75} x seeds {42,43}.
# ================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM192=gdm_tworoom_s10_gc.pt
GDM576=gdm_tworoom_s10_gc_paired.pt
LENS=readout_tworoom_paired_h10.pt
test -f "$GDM576" || { echo "FAILED: paired drafter missing (run the prep job)"; exit 1; }
test -f "$LENS"   || { echo "FAILED: lens missing (run the lens job)"; exit 1; }

ID=$SLURM_ARRAY_TASK_ID
CELL=$((ID % 4))                       # horizon x seed
OFF=$([ $((CELL / 2)) -eq 0 ] && echo 25 || echo 75)
SEED=$((42 + CELL % 2))
BUD=$((2 * OFF))
MODE=$([ "$OFF" -le 25 ] && echo short || echo long)

case $((ID / 4)) in
  0) ARM=baseline;   NAME=flat;          EXTRA=() ;;
  1) ARM=gdm;        NAME=gdm;           EXTRA=(--gdm-ckpt "$GDM192" --gdm-steps 50) ;;
  2) ARM=specaccept; NAME=specLeWM_t020; EXTRA=(--gdm-ckpt "$GDM192" --gdm-steps 50 --accept-tau 0.20) ;;
  3) ARM=specpaired; NAME=specPair_t020; EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.20) ;;
  4) ARM=specpaired; NAME=specPair_t098; EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.098) ;;
  # lens tau is NOT the checkpoint's tau_derived (None -- its gap never opened);
  # it is the criterion floor p50 measured IN LENS SPACE by probe_lens_floor
  # (Results/gap_stat/lens_floor_tworoom.json): equiv p50 0.3617, hop10 p50
  # 0.8190, cross p50 1.3202 -- the lens does widen bulk separation (1.66x raw
  # -> 2.26x) even though [p90, p10] stays closed.
  5) ARM=specpaired; NAME=specPair_lens; EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --readout-ckpt "$LENS" --readout-tau 0.3617) ;;
esac

LOG=logs/2room_pair_${NAME}_t${OFF}_seed${SEED}.log
echo "=== tworoom paired battery arm=$NAME (subgoal=$ARM) t=$OFF seed=$SEED ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
  --start final --eval-filter success --require-cross-room --episode-min 4000 \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE arm=$NAME t=$OFF seed=$SEED ==="
