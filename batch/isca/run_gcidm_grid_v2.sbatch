#!/usr/bin/env bash
#SBATCH --job-name=gcidm-grid2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_grid2_%A_%a.log
# ============ GC-IDM GRID v2: EACH CELL SERVED AT ITS OWN H_max = T ==========
# Supersedes job 2329277. Every cell loads the checkpoint trained with
# H_max = T = 2t for THAT horizon (run_gcidm_hmax_fix.sbatch), so the horizon
# signal is never clamped outside its training range.
# ONE SEED PER CELL, deliberately: v1's 8 Reacher / 4 PushT seeds came back
# byte-identical (deterministic MLP, pinned starts, deterministic dynamics),
# so extra seeds are replicas, not samples. Cube keeps its 8 (fresh population
# per seed). Populations are the banked ones, unchanged, so v2 is directly
# comparable to v1 and to our own arms cell for cell.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ENVS=(pusht pusht pusht pusht pusht reacher reacher reacher reacher cube)
OFFS=(25    50    75    100   150   25      50      100     150     150)
ID=$SLURM_ARRAY_TASK_ID
E=${ENVS[$ID]}; OFF=${OFFS[$ID]}; BUD=$((2 * OFF)); HMAX=$BUD
CKPT=gcidm_${E}_h${HMAX}.pt

if [ ! -f "$CKPT" ]; then
  echo "MISSING $CKPT -- run_gcidm_hmax_fix.sbatch must complete first"; exit 1
fi

case $E in
  pusht)
    EPFILE=pusht.episodes150s5as${OFF}.json
    if [ "$OFF" -eq 150 ]; then EPFILE=pusht.episodes150s5.json; fi
    N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
    echo "=== grid2 pusht t=$OFF n=$N budget=$BUD H_max=$HMAX ==="
    python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long --episodes-file $EPFILE \
      --goal-offset ${OFF} --eval-budget ${BUD} --num-eval $N \
      --score block --angles 20 5 --seed 42 --cem-seed 42 \
      2>&1 | tee logs/gcidm2_grid_pusht_t${OFF}_seed42.log
    ;;
  reacher)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
    EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
    N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
    echo "=== grid2 reacher t=$OFF n=$N budget=$BUD H_max=$HMAX ==="
    python -m specaccept.envs.reacher.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --num-eval $N --seed 42 --cem-seed 42 --episodes-file $EPFILE \
      --goal-offset ${OFF} --eval-budget ${BUD} \
      2>&1 | tee logs/gcidm2_grid_reacher_t${OFF}_seed42.log
    ;;
  cube)
    # cube samples a fresh population per seed, so its seeds are real replicates
    for S in 42 43 44 45 46 47 48 49; do
      echo "=== grid2 cube t=150 seed=$S budget=$BUD H_max=$HMAX ==="
      python -m specaccept.envs.cube.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
        --source pretrained --encoder-id quentinll/lewm-cube \
        --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
        --start final --goal-offset 150 --eval-budget ${BUD} \
        --num-eval 128 --seed $S --cem-seed $S --episode-min 8000 \
        2>&1 | tee logs/gcidm2_grid_cube_t150_seed${S}.log
    done
    ;;
esac
echo "=== DONE task $ID ==="
