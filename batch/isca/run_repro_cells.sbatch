#!/usr/bin/env bash
#SBATCH --job-name=repro-cells
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/repro_cells_%A_%a.log
# ==== REPRO SMOKE stage 2: one NO-QUOTE cell per env, fresh seed 60 ====
# docs/repro_smoke_prereg.md (registered first): plumbing test only; SRs
# non-quotable; PASS = completes + within +/-8pp of the banked family ref
# (pusht ~61.5 / reacher ~94 / cube ~79.9 / tworoom ~57). Runs in the
# freshly built .venv_repro (stage 1), production venv untouched.
# ======================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv_repro/bin/activate
mkdir -p logs
SEED=60

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== repro pusht spec t=150 seed=$SEED (ref ~61.5, unfiltered pop) ==="
    python -m specaccept.envs.pusht.eval --subgoal specaccept \
      --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long \
      --goal-offset 150 --eval-budget 300 --num-eval 64 --score block --angles 20 \
      --seed "$SEED" --cem-seed "$SEED" --horizon 2 --receding-horizon 2 \
      2>&1 | tee logs/repro_pusht_seed${SEED}.log
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
      --out reacher_repro.ep.json --n 64 --max-offset 150 --episode-min 8000 --seed "$SEED"
    echo "=== repro reacher spec t=150 seed=$SEED (ref ~94) ==="
    python -m specaccept.envs.reacher.eval --subgoal specaccept \
      --gdm-ckpt gdm_reacher_s10.pt --accept-tau 0.20 --gdm-steps 8 \
      --horizon 2 --receding-horizon 2 \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --episodes-file reacher_repro.ep.json \
      --goal-offset 150 --eval-budget 300 \
      2>&1 | tee logs/repro_reacher_seed${SEED}.log
    ;;
  2)
    echo "=== repro cube champion seed=$SEED (ref ~79.9) ==="
    python -m specaccept.envs.cube.eval --subgoal specaccept \
      --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate --accept-tau 0.20 --gdm-steps 3 \
      --horizon 2 --receding-horizon 2 \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
      --num-samples 600 --n-steps 30 \
      2>&1 | tee logs/repro_cube_seed${SEED}.log
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    echo "=== repro tworoom spec t=75 seed=$SEED (ref ~57) ==="
    python -m specaccept.envs.tworoom.eval --subgoal specpaired \
      --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt --gdm-steps 50 --accept-tau 0.098 \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode long --goal-offset 75 --eval-budget 150 \
      --eval-filter none --episode-min 4000 --goal-from-proprio \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon 2 \
      2>&1 | tee logs/repro_tworoom_seed${SEED}.log
    ;;
esac
echo "=== DONE repro cell $SLURM_ARRAY_TASK_ID ==="
