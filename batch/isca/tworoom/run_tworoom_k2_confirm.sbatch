#!/usr/bin/env bash
#SBATCH --job-name=2room-k2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_k2_%A_%a.log
# ===== TWOROOM: CONFIRM RUN AT THE DERIVED k*=2 (probe_k, faith-prep 2306153) =====
# The k derivation predicts sampling steps do not change accept decisions
# (bias-to-k50 p50 0.056 < floor 0.098 at k=2). BAR, frozen now: the k=2 arm is
# within 3pp of the k=50 12-seed mean (57.03) at six seeds. Pass = the faithful
# row runs at 25x fewer diffusion steps per draft; fail = k derivation refuted
# on this substrate, recorded, and the row keeps k=50.
# ==================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

SEED=$((42 + SLURM_ARRAY_TASK_ID))
LOG=logs/2room_k2_pair_t098_k2_seed${SEED}.log
echo "=== k*=2 confirm seed=$SEED : anchored protocol RH=2 ==="
python -m specaccept.envs.tworoom.eval --subgoal specpaired \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt --gdm-steps 2 --accept-tau 0.098 \
  --device cuda --mode long --goal-offset 75 --eval-budget 150 \
  --eval-filter none --episode-min 4000 --goal-from-proprio \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon 2 \
  2>&1 | tee "$LOG"
echo "=== DONE k2 seed=$SEED ==="
