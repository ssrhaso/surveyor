#!/usr/bin/env bash
#SBATCH --job-name=gcidm-2room
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_2room_%j.log
# ====== P-2R: GC-IDM ON TWO-ROOM (routing rule's fourth-env prospective test) =
# prereg/2026-08-09_cube_executor_and_tworoom_gcidm.md section B, frozen before
# submission. One sequential job: stride-1 dense cache (filters OFF, first 4000
# episodes only -> disjoint from the eval holdout at episode-min 4000), train
# h50 + h150 at their optimizer spec, then 12-seed cells at their-protocol t=25
# and anchored t=75. References: t=25 flat 83.7; t=75 flat 41.7 / spec 57.0 /
# oracle 56.9. Frozen: GC-IDM t=75 < 57.0 AND t=75 < its own t=25.
# Falsifier: t=75 >= 57.0 (full prominence; Two-Room headline gains a context
# row). Smokes never quoted.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
echo "[data] $H5"
CACHE=subgoals_tworoom_dense_train.pt

if [ ! -f "$CACHE" ]; then
  echo "=== stage 1: stride-1 dense cache (episodes 0-3999, filters OFF) ==="
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 "$H5" --out "$CACHE" \
    --source pretrained --encoder-id quentinll/lewm-tworooms \
    --stride 1 --device cuda --batch-size 256 \
    --no-require-cross-room --no-success-filter \
    --max-episodes 4000 --sample-mode head \
    2>&1 | tee logs/gcidm_2room_cache.log
fi

for H in 50 150; do
  OUT=gcidm_tworoom_h${H}.pt
  if [ ! -f "$OUT" ]; then
    echo "=== stage 2: train GC-IDM H_max=$H (their spec verbatim, seed 0) ==="
    python -m specaccept.train_gcidm \
      --latents "$CACHE" --h5 "$H5" --out "$OUT" \
      --h-max ${H} --hidden 512 --depth 3 --dropout 0.1 \
      --epochs 50 --batch-size 1024 --lr 1e-3 --weight-decay 1e-4 \
      --device cuda --seed 0 \
      2>&1 | tee logs/gcidm_2room_train_h${H}.log
  fi
done

echo "=== stage 3 smoke: their-protocol n=16 (plumbing gate; never quoted) ==="
python -m specaccept.envs.tworoom.eval --subgoal gcidm \
  --gcidm-ckpt gcidm_tworoom_h50.pt \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode short --start random --goal-offset 25 --eval-budget 50 \
  --eval-filter none --episode-min 4000 --goal-from-proprio \
  --num-eval 16 --seed 42 --cem-seed 42 \
  2>&1 | tee logs/gcidm_2room_smoke.log
grep -q "SR" logs/gcidm_2room_smoke.log || { echo "SMOKE FAILED"; exit 1; }

for SEED in 42 43 44 45 46 47 48 49 50 51 52 53; do
  echo "=== P-2R their-protocol t=25 seed=$SEED (h50) ==="
  python -m specaccept.envs.tworoom.eval --subgoal gcidm \
    --gcidm-ckpt gcidm_tworoom_h50.pt \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --mode short --start random --goal-offset 25 --eval-budget 50 \
    --eval-filter none --episode-min 4000 --goal-from-proprio \
    --num-eval 64 --seed $SEED --cem-seed $SEED \
    2>&1 | tee logs/gcidm_2room_t25_seed${SEED}.log
done

for SEED in 42 43 44 45 46 47 48 49 50 51 52 53; do
  echo "=== P-2R anchored t=75 seed=$SEED (h150) ==="
  python -m specaccept.envs.tworoom.eval --subgoal gcidm \
    --gcidm-ckpt gcidm_tworoom_h150.pt \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
    --device cuda --mode long --goal-offset 75 --eval-budget 150 \
    --eval-filter none --episode-min 4000 --goal-from-proprio \
    --num-eval 64 --seed $SEED --cem-seed $SEED \
    2>&1 | tee logs/gcidm_2room_t75_seed${SEED}.log
done
echo "=== gcidm-2room ALL DONE ==="
