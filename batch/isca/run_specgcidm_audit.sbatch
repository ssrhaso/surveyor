#!/usr/bin/env bash
#SBATCH --job-name=sgc-audit
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/sgc_audit_%A_%a.log
# ================== EXECUTOR-RESULT AUDIT (P-EXEC-1 = 96.09) =================
# Before the specgcidm result is quoted anywhere, the same treatment every
# headline got. All on the identical t=150 cell (pusht.episodes150s5.json,
# n=256, budget 300, executor gcidm_pusht_h50.pt, drafter gdm_stride10.pt).
#
#  task 0  REPEAT, byte-identical config to the banked seed-42 cell.
#          Determinism check: any drift from 246/256 = hidden nondeterminism,
#          audit fails, result quarantined.
#  task 1  BLIND COMMIT (tau=999): consume the block with verification off.
#          POST-HOC mechanism control (not pre-registered, labeled as such):
#          if blind ties 96, the verifier is not load-bearing here and the
#          "verifier compensates for the sloppier executor" sentence DIES;
#          if blind collapses, the accept rule carries the result.
#  task 2  EVERY-STEP (tau=0): re-draft at every boundary, no consumption.
#          Isolates what speculative consumption saves at matched executor.
#  task 3  CEM-EXECUTOR spec arm, --time-instrument, same cell: the honest
#          same-cell wall-clock denominator for P-EXEC-2 (my cross-cell
#          0.18s-vs-1.9s approximation is not printable).
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
EPFILE=pusht.episodes150s5.json
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
COMMON=(--source pretrained --h5 "$H5" --device cuda --mode long
        --episodes-file "$EPFILE" --goal-offset 150 --eval-budget 300
        --num-eval "$N" --score block --angles 20 5 --seed 42 --cem-seed 42)

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== AUDIT repeat: byte-identical rerun of banked seed-42 cell ==="
    python -m specaccept.envs.pusht.eval --subgoal specgcidm \
      --gcidm-ckpt gcidm_pusht_h50.pt --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
      --accept-tau 0.20 --sg-steps 10 "${COMMON[@]}" \
      2>&1 | tee logs/sgc_audit_repeat.log
    ;;
  1)
    echo "=== AUDIT blind commit (tau=999, post-hoc mechanism control) ==="
    python -m specaccept.envs.pusht.eval --subgoal specgcidm \
      --gcidm-ckpt gcidm_pusht_h50.pt --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
      --accept-tau 999 --sg-steps 10 "${COMMON[@]}" \
      2>&1 | tee logs/sgc_audit_blind.log
    ;;
  2)
    echo "=== AUDIT every-step (tau=0, consumption off) ==="
    python -m specaccept.envs.pusht.eval --subgoal specgcidm \
      --gcidm-ckpt gcidm_pusht_h50.pt --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
      --accept-tau 0 --sg-steps 10 "${COMMON[@]}" \
      2>&1 | tee logs/sgc_audit_everystep.log
    ;;
  3)
    echo "=== AUDIT CEM-executor spec arm, same cell, timing on ==="
    python -m specaccept.envs.pusht.eval --subgoal specaccept \
      --gdm-ckpt gdm_stride10.pt --gdm-steps 3 --accept-tau 0.20 \
      --time-instrument "${COMMON[@]}" \
      2>&1 | tee logs/sgc_audit_cemtiming.log
    ;;
esac
echo "=== AUDIT task $SLURM_ARRAY_TASK_ID DONE ==="
