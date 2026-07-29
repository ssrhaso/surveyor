#!/usr/bin/env bash
#SBATCH --job-name=poster-strips
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/home/ha676/le-wm/logs/poster_strips_%A_%a.log
# ============== POSTER QUALITATIVE FILMSTRIPS (due Aug 11, code-bound) ==============
# Captures, at every replan boundary, the raw frame the verifier saw plus its
# decision there (advance / redraft / gate), for filmstrip rendering with
# specaccept/render_strips.py. Small runs (n=8, one seed) at the paper cells:
#   0  pusht  spec-accept  t=150 (the headline horizon-extension cell)
#   1  pusht  flat         same protocol, via lerp 1.0 == instrumented-flat
#   2  tworoom specpaired  anchored protocol t=75 RH=2 (also an early smoke
#      test of the Stage B serving path: goal_proprio + wants_frames + events)
# Frames are a QUALITATIVE illustration; no number from these n=8 runs may be
# quoted anywhere.
# ====================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre/home/ha676/le-wm
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p strips

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== pusht spec-accept t=150 strip ==="
    # NOTE: no --episodes-file here; it pins num_eval to its 256 pairs and the
    # strip run is n=8 qualitative (no number from it is quotable anyway)
    python -m specaccept.envs.pusht.eval --subgoal specaccept \
      --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long \
      --goal-offset 150 --eval-budget 300 --num-eval 8 --score block --angles 20 \
      --seed 42 --cem-seed 42 --horizon 2 --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_pusht_spec.log
    ;;
  1)
    echo "=== pusht instrumented-flat (lerp 1.0) t=150 strip ==="
    python -m specaccept.envs.pusht.eval --subgoal lerp --lerp-frac 1.0 \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long \
      --goal-offset 150 --eval-budget 300 --num-eval 8 --score block --angles 20 \
      --seed 42 --cem-seed 42 --horizon 2 --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_pusht_flat.log
    ;;
  2)
    echo "=== tworoom specpaired anchored t=75 RH=2 strip ==="
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.envs.tworoom.eval --subgoal specpaired \
      --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt --gdm-steps 50 --accept-tau 0.098 \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode long --goal-offset 75 --eval-budget 150 \
      --eval-filter none --episode-min 4000 --goal-from-proprio \
      --num-eval 8 --seed 42 --cem-seed 42 --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip_tworoom_spec.log
    ;;
esac
echo "=== STRIP TASK $SLURM_ARRAY_TASK_ID DONE ==="
