#!/usr/bin/env bash
#SBATCH --job-name=gcidm-anchor
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_anchor_%A_%a.log
# ============= GC-IDM ANCHOR: REPRODUCE THE PUBLISHED CELLS FIRST ============
# The cube lesson: anchor before sweeping. No GC-IDM cell of ours may be quoted
# until this array is read.
# TARGETS (arXiv 2605.08732 Table 1, n=50 / n=200, 3 seeds):
#     Two-Room  100.0 / 100.0      Push-T  84.7 / 84.2
#     OGB-Cube   99.3 /  98.7      Reacher 100.0 / 99.7
# Their protocol = LeWM's eval config verbatim: goal offset 25, budget T=50,
# H_max=T=50, closed loop, one action per step. That is what runs here.
# ONE DECLARED DEVIATION: their protocol samples starts from the WHOLE dataset;
# this runs their horizon and budget on our HELD-OUT populations (reacher/cube
# episodes < 8000; pusht eval-population episodes excluded by name), because a
# comparator evaluated on its own training episodes would be meaningless. The
# deviation handicaps the baseline relative to its paper, never flatters it.
# Two-Room is absent: no stride-1 dense latent cache here, so no GC-IDM was
# trained for it. Recorded as missing rather than worked around.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ENVS=(pusht reacher cube)
E=${ENVS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((42 + SLURM_ARRAY_TASK_ID % 3))

case $E in
  pusht)
    EPFILE=pusht.episodes150s5as25.json
    N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
    echo "=== gcidm-anchor pusht t=25 budget=50 n=$N seed=$SEED (target 84.7) ==="
    python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt gcidm_pusht.pt \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long --episodes-file $EPFILE \
      --goal-offset 25 --eval-budget 50 --num-eval $N \
      --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
      2>&1 | tee logs/gcidm_anchor_pusht_seed${SEED}.log
    ;;
  reacher)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    echo "=== gcidm-anchor reacher t=25 budget=50 n=50 seed=$SEED (target 100.0) ==="
    python -m specaccept.envs.reacher.eval --subgoal gcidm --gcidm-ckpt gcidm_reacher.pt \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --start random --goal-offset 25 --eval-budget 50 --num-eval 50 \
      --episode-min 8000 --seed $SEED --cem-seed $SEED \
      2>&1 | tee logs/gcidm_anchor_reacher_seed${SEED}.log
    ;;
  cube)
    echo "=== gcidm-anchor cube t=25 budget=50 n=50 seed=$SEED (target 99.3) ==="
    # NOTE: this is LeWM's Cube protocol, the one P-DISP-1 showed is 38.38%
    # vacuous. A high number here is EXPECTED and is not evidence of capability;
    # it anchors our implementation against theirs, nothing more.
    python -m specaccept.envs.cube.eval --subgoal gcidm --gcidm-ckpt gcidm_cube.pt \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start random --goal-offset 25 --eval-budget 50 --num-eval 50 \
      --episode-min 8000 --seed $SEED --cem-seed $SEED \
      2>&1 | tee logs/gcidm_anchor_cube_seed${SEED}.log
    ;;
esac
echo "=== GCIDM ANCHOR DONE $E seed=$SEED ==="
