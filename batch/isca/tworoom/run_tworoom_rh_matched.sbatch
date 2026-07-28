#!/usr/bin/env bash
#SBATCH --job-name=2room-rhmatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_rhmatch_%A_%a.log
# ============ TWOROOM: REPLAN-RATE-MATCHED DRAFTING (the fair comparison) ============
# The 2300264 sweep found flat's strongest replan rate, and it is NOT the one the
# drafting arms ran at:
#     flat   t=25  RH1 60.94 | RH2 64.84 | RH5 64.06
#     flat   t=75  RH1 44.53 | RH2 42.19 | RH5 27.34
# Every drafting arm in 2300235 ran at the default RH=5. So the reported margin
# compared flat-at-its-best against drafting-at-one-arbitrary-config. This runs
# the drafting arms at RH 1 and 2 so both sides are at their best.
#
# FROZEN PREDICTION (before this ran). TwoRoom's horizon collapse is REDUCIBLE by
# a cheaper flat configuration: RH=1 alone recovers t=75 from 27.34 to 44.53. On
# PushT no flat config recovers it (RH1 88->18, RH2 96->47) and drafting wins
# +45pp there. The proposed scope condition is that drafting pays only when the
# collapse survives the strongest flat config. It therefore predicts:
#   R1: drafting at RH=1 does NOT beat flat at RH=1 (44.53) at t=75 by more
#       than noise, i.e. margin < +5pp.
# If drafting at RH=1 jumps well above 44.53, the scope condition is WRONG and
# TwoRoom has real headroom after all. That is the outcome worth having.
#
# Arms use the REPAIRED drafters only (gate 2300079):
#   gdm / specaccept  -> gdm_tworoom_s10_gc_e400v.pt        ratio 0.989
#   specpaired        -> gdm_tworoom_s10_gc_paired_e400v.pt ratio 1.014 / 1.001
# tau: LeWM-space arm 0.20 (transfer default); paired arm 0.098, the band floor
# that was the only fixed tau to clear flat at both horizons in 2300235.
#
# Array order puts the decisive t=75 cells first.
# ====================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM192=gdm_tworoom_s10_gc_e400v.pt
GDM576=gdm_tworoom_s10_gc_paired_e400v.pt
test -f "$GDM192"; test -f "$GDM576"

ID=$SLURM_ARRAY_TASK_ID
BLOCK=$((ID / 6))                 # 0: t75 RH1 | 1: t75 RH2 | 2: t25 RH1 | 3: t25 RH2
OFF=$([ $BLOCK -le 1 ] && echo 75 || echo 25)
RH=$([ $((BLOCK % 2)) -eq 0 ] && echo 1 || echo 2)
SUB=$((ID % 6))
SEED=$((42 + SUB % 2))
BUD=$((2 * OFF))
MODE=$([ "$OFF" -le 25 ] && echo short || echo long)

case $((SUB / 2)) in
  0) ARM=gdm;        NAME=gdm;            EXTRA=(--gdm-ckpt "$GDM192" --gdm-steps 50) ;;
  1) ARM=specaccept; NAME=specLeWM_t020;  EXTRA=(--gdm-ckpt "$GDM192" --gdm-steps 50 --accept-tau 0.20) ;;
  2) ARM=specpaired; NAME=specPair_t098;  EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.098) ;;
esac

LOG=logs/2room_rh${RH}_${NAME}_t${OFF}_seed${SEED}.log
echo "=== tworoom RH-matched arm=$NAME RH=$RH t=$OFF seed=$SEED ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
  --start final --eval-filter success --require-cross-room --episode-min 4000 \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon "$RH" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE arm=$NAME RH=$RH t=$OFF seed=$SEED ==="
