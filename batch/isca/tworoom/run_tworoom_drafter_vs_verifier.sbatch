#!/usr/bin/env bash
#SBATCH --job-name=2room-dvv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-47
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_dvv_%A_%a.log
# ========== TWOROOM: IS THE GAIN THE DRAFTER OR THE VERIFIER? ==========
# In 2300331 the paired arm beat every-step drafting in all four cells
# (+7.0, +7.0, +18.0, +7.8) while its call_ratio sat at 0.95-0.96, i.e. the
# verifier accepted about 4 percent of the time and was doing almost nothing.
# The two arms also differ in WHICH DRAFTER they use: 576-d paired versus
# 192-d plain. So the gain is probably the drafter, not the verification, and
# that is a different claim entirely.
#
# The isolation needs no new code. SpecAcceptPairedSource with accept_tau=0
# never verifies, so it re-drafts at every replan and serves position m+1:
# that is every-step drafting with the PAIRED drafter. Three points on one
# axis, same drafter, only the consumption policy changing:
#   tau = 0      every-step, no verification   (call_ratio -> 1.00)
#   tau = 0.098  the real arm
#   tau = 999    blind chaining, never re-drafts on rejection
# plus the 192-d every-step arm as the drafter contrast, and flat as anchor.
#
# HYPOTHESIS UNDER TEST: if tau=0 matches tau=0.098, the verifier contributes
# nothing on TwoRoom and the entire effect is that training the drafter to
# jointly predict a well-structured auxiliary representation (DINOv2) improves
# its predictions in the planner's own space. That is an auxiliary-objective
# result, not a spec-accept result, and it must be reported as such.
#
# SIX seeds, because 2300331 had cells whose two seeds differed by 25pp and no
# margin there was interpretable.
#
# Cells are the two where the paired arm looked strongest:
#   t=25 RH=1  (flat 60.94)      t=75 RH=2  (flat 42.19)
# Strongest flat overall for reference: t=25 64.84 (RH2), t=75 44.53 (RH1).
# =======================================================================
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
CELL=$((ID / 24))
if [ $CELL -eq 0 ]; then OFF=25; RH=1; else OFF=75; RH=2; fi
REST=$((ID % 24))
SEED=$((42 + REST % 6))
BUD=$((2 * OFF))
MODE=$([ "$OFF" -le 25 ] && echo short || echo long)

case $((REST / 6)) in
  0) ARM=baseline;   NAME=flat;           EXTRA=() ;;
  1) ARM=gdm;        NAME=gdm192;         EXTRA=(--gdm-ckpt "$GDM192" --gdm-steps 50) ;;
  2) ARM=specpaired; NAME=pair_everystep; EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.0) ;;
  3) ARM=specpaired; NAME=pair_t098;      EXTRA=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.098) ;;
esac

LOG=logs/2room_dvv_${NAME}_t${OFF}_rh${RH}_seed${SEED}.log
echo "=== arm=$NAME t=$OFF RH=$RH seed=$SEED ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
  --start final --eval-filter success --require-cross-room --episode-min 4000 \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon "$RH" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE arm=$NAME t=$OFF RH=$RH seed=$SEED ==="
