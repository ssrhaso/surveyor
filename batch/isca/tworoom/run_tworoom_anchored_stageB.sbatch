#!/usr/bin/env bash
#SBATCH --job-name=2room-anchB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-29
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_anchB_%A_%a.log
# ========== TWOROOM STAGE B: PAIR ARMS AT THE ANCHORED PROTOCOL ==========
# Bars frozen in docs/tworoom_paired_prereg.md (07-29 amendment + Stage-A
# finalization) BEFORE any cell here ran. Stage A measured: flat RH2 42.71
# (strongest), oracle 57.81 -> gate PASSED (+15.1 >= +3). k*=8 from the bok
# probe; rule 'goal' primary ('feas' = stay-put risk, exploratory only).
# Spec arms at RH=2 (previously-selected serving config, not re-tuned).
#   B1: pair+verify t098 >= 46.71 | B2: +bok >= B1arm+2 | B4: nofilt >= B1arm+2
# =========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM=gdm_tworoom_s10_gc_paired_e400v.pt
GDMNF=gdm_tworoom_s10_gc_paired_nofilt_e400v.pt
ANCHORED=(--mode long --goal-offset 75 --eval-budget 150 --eval-filter none
          --episode-min 4000 --goal-from-proprio --num-eval 64
          --receding-horizon 2 --gdm-steps 50)

ID=$SLURM_ARRAY_TASK_ID
case $ID in
  [0-5])    SEED=$((42 + ID));      NAME=pair_noverify
            EXTRA=(--gdm-ckpt "$GDM" --accept-tau 0.0) ;;
  [6-9]|1[01]) SEED=$((42 + ID - 6)); NAME=pair_t098
            EXTRA=(--gdm-ckpt "$GDM" --accept-tau 0.098) ;;
  1[2-7])   SEED=$((42 + ID - 12)); NAME=pair_t098_bok8goal
            EXTRA=(--gdm-ckpt "$GDM" --accept-tau 0.098
                   --best-of-k 8 --bok-score goal) ;;
  1[89]|20) SEED=$((42 + ID - 18)); NAME=pair_t020
            EXTRA=(--gdm-ckpt "$GDM" --accept-tau 0.20) ;;
  2[1-6])   SEED=$((42 + ID - 21)); NAME=pair_nofilt_t098
            EXTRA=(--gdm-ckpt "$GDMNF" --accept-tau 0.098) ;;
  2[7-9])   SEED=$((42 + ID - 27)); NAME=pair_t098_bok8feas
            EXTRA=(--gdm-ckpt "$GDM" --accept-tau 0.098
                   --best-of-k 8 --bok-score feas) ;;
esac

LOG=logs/2room_anchB_${NAME}_seed${SEED}.log
echo "=== Stage B arm=$NAME seed=$SEED : anchored protocol RH=2 ==="
python -m specaccept.envs.tworoom.eval --subgoal specpaired \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  "${ANCHORED[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE $NAME seed=$SEED ==="
