#!/usr/bin/env bash
#SBATCH --job-name=gcidm-grid
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-59%12
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_grid_%A_%a.log
# ================ GC-IDM ON OUR GRID (anchor passed 2026-08-07) =============
# Gate cleared: the anchor (job 2329266) reproduced the published cells --
# Reacher 100.00 vs 100.0, PushT 100.00 vs 84.7, Cube 92.67 vs 99.3, at 1.55M
# params vs their ~1.5M. No cell underperformed its paper, so it is quotable.
# Same frozen checkpoints on OUR populations, matched cell for cell to the
# random floor and headline grid: identical episode files, budget 2t, each
# env's own criterion, seeds matched for paired margins. Reacher populations
# stay unspliced (max-offset-100 for t<=100, max-offset-150 for t=150).
#   PushT   t{25,50,75,100,150} x seeds 42-45   (20)
#   Reacher t{25,50,100,150}    x seeds 42-49   (32)
#   Cube    t150                x seeds 42-49    (8)
# Frozen in prereg/2026-08-07_baselines.md:
#   P-GCIDM-1: wherever GC-IDM beats LeWM-flat it BECOMES the strongest-flat
#              reference and the margin row is recomputed against it, even
#              where that shrinks our margin.
#   P-GCIDM-2: GC-IDM collapses at long horizon like flat CEM (monotone fall
#              on PushT, >=40pp below its own t=25 cell at t=150); if it does
#              NOT, that is headline-affecting and reported prominently.
# H_max=50 is the paper's own rule (H_max = T); at t>50 the normalised horizon
# saturates at 1.0 -- their design, and the mechanism P-GCIDM-2 predicts.
# Decisions per episode are printed so cost travels with success.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ID=$SLURM_ARRAY_TASK_ID

if [ "$ID" -lt 20 ]; then
  OFFSETS=(25 50 75 100 150)
  OFF=${OFFSETS[$((ID / 4))]}
  SEED=$((42 + ID % 4))
  EPFILE=pusht.episodes150s5as${OFF}.json
  if [ "$OFF" -eq 150 ]; then EPFILE=pusht.episodes150s5.json; fi
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== gcidm-grid pusht t=$OFF seed=$SEED n=$N (budget=$((2*OFF))) ==="
  python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt gcidm_pusht.pt \
    --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
    --device cuda --mode long --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) --num-eval $N \
    --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
    2>&1 | tee logs/gcidm_grid_pusht_t${OFF}_seed${SEED}.log

elif [ "$ID" -lt 52 ]; then
  R=$((ID - 20))
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  OFFSETS=(25 50 100 150)
  OFF=${OFFSETS[$((R / 8))]}
  SEED=$((42 + R % 8))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== gcidm-grid reacher t=$OFF seed=$SEED n=$N (budget=$((2*OFF))) ==="
  python -m specaccept.envs.reacher.eval --subgoal gcidm --gcidm-ckpt gcidm_reacher.pt \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval $N --seed ${SEED} --cem-seed ${SEED} --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    2>&1 | tee logs/gcidm_grid_reacher_t${OFF}_seed${SEED}.log

else
  SEED=$((42 + ID - 52))
  echo "=== gcidm-grid cube t=150 seed=$SEED (budget=300) ==="
  python -m specaccept.envs.cube.eval --subgoal gcidm --gcidm-ckpt gcidm_cube.pt \
    --source pretrained --encoder-id quentinll/lewm-cube \
    --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
    --start final --goal-offset 150 --eval-budget 300 \
    --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
    2>&1 | tee logs/gcidm_grid_cube_t150_seed${SEED}.log
fi
echo "=== DONE task $ID ==="
