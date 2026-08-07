#!/usr/bin/env bash
#SBATCH --job-name=gcidm-power
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_power_%A_%a.log
# ============ GC-IDM STATISTICAL POWER: MORE EPISODES, NOT MORE SEEDS ========
# The grid (2329277) ran Reacher at 8 seeds / PushT at 4 and the harvester
# pooled them as 1024 episodes/cell. Wrong: GC-IDM is a DETERMINISTIC MLP on
# pinned starts and deterministic dynamics, so every seed reproduces the same
# rollouts (reacher [128]x8 at every t -> 1 distinct; pusht t150 -> 2). The
# effective sample was 128 (Reacher) / 256 (PushT), and more seeds would only
# tighten the error bar falsely. The only lever is MORE DISTINCT EPISODES:
# build larger held-out populations and run once each (seed recorded anyway).
#   reacher: 512 episodes (4x) at t{25,50,100,150}
#   pusht:   512 episodes (2x) at t150, the cell carrying the headline margin
# Cube is NOT re-run: fresh population per seed, so its 8 seeds are genuine
# replicates (spread 109-119) and its ~1024 episodes are real.
# FAIRNESS: our own arm is stochastic, so its 8 seeds ARE real replicates on
# the 128-episode file; ours-on-128 vs GC-IDM-on-512 is not paired. This only
# tightens GC-IDM's population estimate; a paired margin at larger n needs OUR
# arm re-run on the same file (expensive CEM, separate decision).
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ID=$SLURM_ARRAY_TASK_ID
H5R=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

if [ "$ID" -lt 4 ]; then
  OFFSETS=(25 50 100 150)
  OFF=${OFFSETS[$ID]}
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_power512.ep${MAXOFF}.json
  if [ ! -f "$EPFILE" ]; then
    python batch/build_reacher_horizon_episodes.py --h5 "$H5R" --out "$EPFILE" \
      --n 512 --max-offset ${MAXOFF} --episode-min 8000 --seed 999
  fi
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== gcidm-power reacher t=$OFF n=$N (deterministic arm, 1 seed) ==="
  python -m specaccept.envs.reacher.eval --subgoal gcidm --gcidm-ckpt gcidm_reacher.pt \
    --source local --local-dir encoder_reacher --h5 "$H5R" --device cuda \
    --num-eval $N --seed 42 --cem-seed 42 --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    2>&1 | tee logs/gcidmpow_reacher_t${OFF}_seed42.log
else
  EPFILE=pusht_power512.episodes150.json
  if [ ! -f "$EPFILE" ]; then
    python - <<'PY'
import json
from specaccept.envs.pusht.eval import sample_long
H5 = "/lustre/home/ha676/data/pusht/pusht_expert_train.h5"
eps, starts = sample_long(H5, 512, 150, 999)
json.dump({"episodes": [[int(e), int(s)] for e, s in zip(eps, starts)],
           "goal_offset": 150, "seed": 999},
          open("pusht_power512.episodes150.json", "w"))
print(f"[build] {len(eps)} episodes -> pusht_power512.episodes150.json")
PY
  fi
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== gcidm-power pusht t=150 n=$N (deterministic arm, 1 seed) ==="
  python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt gcidm_pusht.pt \
    --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
    --device cuda --mode long --episodes-file $EPFILE \
    --goal-offset 150 --eval-budget 300 --num-eval $N \
    --score block --angles 20 5 --seed 42 --cem-seed 42 \
    2>&1 | tee logs/gcidmpow_pusht_t150_seed42.log
fi
echo "=== DONE task $ID ==="
