#!/usr/bin/env bash
#SBATCH --job-name=2room-anchExt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_anchExt_%A_%a.log
# ===== TWOROOM: 12-SEED EXTENSION FOR THE B1 HEADLINE (anchored protocol) =====
# Pre-registered extension seeds 48-53 for the two B1 arms (flat RH2 and
# pair+verify tau=0.098). Fired UNCONDITIONALLY, before any Stage B cell was
# read: this only adds data and removes a conditional branch; the pooled
# 12-seed bar (margin >= +3pp AND t >= 2) applies exactly as frozen in
# docs/tworoom_paired_prereg.md. Seeds 42-47 come from Stage A (flat) and
# Stage B (pair); these pool directly, same protocol.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((48 + ID % 6))
ANCHORED=(--mode long --goal-offset 75 --eval-budget 150 --eval-filter none
          --episode-min 4000 --goal-from-proprio --num-eval 64
          --receding-horizon 2)

if [ "$ID" -lt 6 ]; then
  NAME=flat_rh2
  ARGS=(--subgoal baseline)
else
  NAME=pair_t098
  ARGS=(--subgoal specpaired --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt
        --gdm-steps 50 --accept-tau 0.098)
fi

LOG=logs/2room_anchExt_${NAME}_seed${SEED}.log
echo "=== extension arm=$NAME seed=$SEED : anchored protocol RH=2 ==="
python -m specaccept.envs.tworoom.eval \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  "${ANCHORED[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "=== DONE $NAME seed=$SEED ==="
