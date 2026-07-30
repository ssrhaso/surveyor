#!/usr/bin/env bash
#SBATCH --job-name=poster-strips2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/poster_strips2_%A_%a.log
# ========= POSTER FILMSTRIPS, ROUND 2: CUBE + REACHER (+ tworoom flat) =========
# Same convention as run_poster_strips.sbatch: n=8 single-seed QUALITATIVE
# captures at the paper cells; NO number from these runs may be quoted anywhere.
# Flat arms are captured via the lerp-1.0 instrumented-flat tautology (the
# fig:cost convention) because the strip plumbing lives in the FFJEPA policy.
#   0  cube    certified spec (gc10 champion: goal-gate, tau=0.20, k=3)
#   1  cube    flat (lerp 1.0, RH5, 600 samples = the certified CEM budget)
#   2  reacher spec t=150 (tau=0.20, k=8, S=10; max-offset-150 population)
#   3  reacher flat (lerp 1.0, RH5) same population
#   4  tworoom flat (lerp 1.0, RH2) anchored t=75 -- pairs with the existing
#      strip_tworoom_specpaired_t75_seed42.npz for a side-by-side panel
# ===============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre/home/ha676/le-wm
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p strips logs

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== cube certified-spec strip (champion config) ==="
    python -m specaccept.envs.cube.eval \
      --subgoal specaccept --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate \
      --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2 \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 8 --seed 42 --cem-seed 42 --episode-min 8000 \
      --num-samples 600 --n-steps 30 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_cube_spec.log
    ;;
  1)
    echo "=== cube instrumented-flat (lerp 1.0) strip ==="
    python -m specaccept.envs.cube.eval \
      --subgoal lerp --lerp-frac 1.0 --horizon 5 --receding-horizon 5 \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 8 --seed 42 --cem-seed 42 --episode-min 8000 \
      --num-samples 600 --n-steps 30 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_cube_flat.log
    ;;
  2|3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    EPFILE=reacher_strip8.ep.json
    python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
      --n 8 --max-offset 150 --episode-min 8000 --seed 42
    if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
      echo "=== reacher spec t=150 strip ==="
      python -m specaccept.envs.reacher.eval \
        --subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt \
        --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2 \
        --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
        --num-eval 8 --seed 42 --cem-seed 42 --episodes-file $EPFILE \
        --goal-offset 150 --eval-budget 300 \
        --dump-strip 8 --dump-strip-out strips \
        2>&1 | tee logs/poster_strip_reacher_spec.log
    else
      echo "=== reacher instrumented-flat (lerp 1.0) t=150 strip ==="
      python -m specaccept.envs.reacher.eval \
        --subgoal lerp --lerp-frac 1.0 --horizon 5 --receding-horizon 5 \
        --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
        --num-eval 8 --seed 42 --cem-seed 42 --episodes-file $EPFILE \
        --goal-offset 150 --eval-budget 300 \
        --dump-strip 8 --dump-strip-out strips \
        2>&1 | tee logs/poster_strip_reacher_flat.log
    fi
    ;;
  4)
    echo "=== tworoom instrumented-flat (lerp 1.0) anchored t=75 strip ==="
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.envs.tworoom.eval --subgoal lerp --lerp-frac 1.0 \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode long --goal-offset 75 --eval-budget 150 \
      --eval-filter none --episode-min 4000 --goal-from-proprio \
      --num-eval 8 --seed 42 --cem-seed 42 --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_tworoom_flat.log
    ;;
esac
echo "=== STRIP2 TASK $SLURM_ARRAY_TASK_ID DONE ==="
