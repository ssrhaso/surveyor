#!/usr/bin/env bash
#SBATCH --job-name=sgc-cube
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/sgc_cube_%A_%a.log
# ============ P-EXEC-8: THE CERTIFIED EXECUTOR ARM ON CUBE ====================
# prereg/2026-08-09_cube_executor_and_tworoom_gcidm.md section A, frozen before
# submission. SURVEYOR-Base w/ GC-IDM executor + arrival gate (gate-only mode,
# no c* probe, no CEM anywhere in the arm): drafter gdm_cube_s10_gc.pt (k=3),
# executor gcidm_cube_h300.pt, tau=0.20, S=10. Seeds 42-49 = the identical
# per-seed populations of the banked plain-GC-IDM grid-v2 cells (pooled 89.8).
# Frozen: executor arm >= 2pp BELOW plain GC-IDM (the tax, fourth measurement).
# Falsifier: meets or beats it. Smoke n=16 gates each task, never quoted.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
S=$((42 + SLURM_ARRAY_TASK_ID))
for f in gcidm_cube_h300.pt gdm_cube_s10_gc.pt; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

echo "=== cube specgcidm SMOKE n=16 (plumbing gate; never quoted) ==="
python -m specaccept.envs.cube.eval --subgoal specgcidm \
  --gcidm-ckpt gcidm_cube_h300.pt --gdm-ckpt gdm_cube_s10_gc.pt --gdm-steps 3 \
  --accept-tau 0.20 --sg-steps 10 --goal-gate \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 16 --seed 42 --cem-seed 42 --episode-min 8000 \
  2>&1 | tee logs/sgc_cube_smoke_task${SLURM_ARRAY_TASK_ID}.log
grep -q "SR = " logs/sgc_cube_smoke_task${SLURM_ARRAY_TASK_ID}.log || { echo "SMOKE FAILED"; exit 1; }

echo "=== P-EXEC-8 cube t=150 seed=$S (population-matched to grid-v2) ==="
python -m specaccept.envs.cube.eval --subgoal specgcidm \
  --gcidm-ckpt gcidm_cube_h300.pt --gdm-ckpt gdm_cube_s10_gc.pt --gdm-steps 3 \
  --accept-tau 0.20 --sg-steps 10 --goal-gate \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed $S --cem-seed $S --episode-min 8000 \
  2>&1 | tee logs/sgc_cube_t150_seed${S}.log
echo "=== sgc-cube seed $S DONE ==="
