#!/usr/bin/env bash
#SBATCH --job-name=random-smoke
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/randsmoke_%A_%a.log
# ================== RANDOM-FLOOR PLUMBING SMOKE (NOT A RESULT) ===============
# One cell per env at the SHORTEST horizon to prove --subgoal random reaches
# swm.policy.RandomPolicy, seeds through set_policy, and scores under each env's
# own criterion. Numbers here are throwaway: the only thing checked is that each
# task prints a [RESULT]/SR line and exits 0.
# NOTE: the episode-file drivers (PushT, Reacher) assert num_eval == len(file),
# so those two cannot be shrunk with --num-eval; N is read off the file exactly
# as the real grid does. Cube/Two-Room sample their populations and do shrink.
# Fire the real grid (run_random_baseline.sbatch, 84 tasks) only after 4/4 pass.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

case $SLURM_ARRAY_TASK_ID in
  0)
    EPFILE=pusht.episodes150s5as25.json
    N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
    echo "=== smoke pusht t=25 n=$N ==="
    python -m specaccept.envs.pusht.eval --subgoal random \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long --episodes-file $EPFILE \
      --goal-offset 25 --eval-budget 50 --num-eval $N \
      --score block --angles 20 5 --seed 42 --cem-seed 42
    ;;
  1)
    EPFILE=reacher_gatev4c.ep100.s999.json
    N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
    echo "=== smoke reacher t=25 n=$N ==="
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.envs.reacher.eval --subgoal random \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --num-eval $N --seed 42 --cem-seed 42 \
      --episodes-file $EPFILE \
      --goal-offset 25 --eval-budget 50
    ;;
  2)
    echo "=== smoke cube t=150 n=16 ==="
    python -m specaccept.envs.cube.eval --subgoal random \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 16 --seed 42 --cem-seed 42 --episode-min 8000
    ;;
  3)
    echo "=== smoke tworoom t=25 n=16 ==="
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.envs.tworoom.eval --subgoal random \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --seed 42 --cem-seed 42 \
      --mode short --start random --goal-offset 25 --eval-budget 50 \
      --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 16
    ;;
esac
echo "=== SMOKE DONE task $SLURM_ARRAY_TASK_ID ==="
