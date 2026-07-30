#!/usr/bin/env bash
#SBATCH --job-name=2room-comp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-29
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_comp_%A_%a.log
# ============ COMPOSITE WINDOW-RULE POLICY, END-TO-END (PRE-REGISTERED) ============
# Bars frozen in docs/tworoom_paired_prereg.md ("COMPOSITE END-TO-END BATTERY",
# 2026-07-30) BEFORE submission. ONE policy at every t: the DRIVER routes on the
# task's known goal distance via --composite-crossover 45 (the measured crossover,
# midpoint of the confirmed [40,50] band -- the one constant). goal_offset <= 45
# -> flat branch; else certified spec (specpaired, tau=0.098 derived, DINOv2
# verify). Branch RH frozen by prior-prereg convention, NOT tuned here:
# their-protocol flat RH5 (their own serving config), anchored flat RH2 (Stage A's
# strongest flat), spec RH2 (headline arm). Protocols = the banked cells verbatim;
# seeds 42-47 (same as banked -> paired, integration-test reading).
#   IDs 0-5: t=25 their-protocol | 6-11: t=40 | 12-17: t=50 | 18-23: t=60 |
#   24-29: t=75 (anchored)
# B-COMP-1: composite >= its branch's banked mean - 1pp at every t
#   (refs: 84.89 / 65.36 / 64.59 / 55.73 / 55.47). Miss = integration fault,
#   reported as-is; no banked number changes under any outcome.
# ===================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM=gdm_tworoom_s10_gc_paired_e400v.pt
SPEC=(--subgoal specpaired --gdm-ckpt "$GDM" --gdm-steps 50 --accept-tau 0.098)
COMP=(--composite-crossover 45 --composite-spec-rh 2)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 6))
TS=(25 40 50 60 75); T=${TS[$((ID / 6))]}

if [ "$T" -eq 25 ]; then
  PROTO=(--mode short --start random --goal-offset 25 --eval-budget 50
         --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)
  FRH=5   # their-protocol serving convention (LeWM's own config)
else
  PROTO=(--mode long --goal-offset "$T" --eval-budget $((2 * T))
         --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)
  FRH=2   # anchored-protocol reference flat (Stage A's strongest)
fi

LOG=logs/2room_composite_t${T}_seed${SEED}.log
echo "=== composite window-rule t=$T seed=$SEED (crossover=45, flat-rh=$FRH) ==="
python -m specaccept.envs.tworoom.eval \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  "${PROTO[@]}" "${SPEC[@]}" "${COMP[@]}" --composite-flat-rh "$FRH" \
  2>&1 | tee "$LOG"
echo "=== DONE composite t=$T seed=$SEED ==="
