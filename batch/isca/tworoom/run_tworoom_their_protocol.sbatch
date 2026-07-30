#!/usr/bin/env bash
#SBATCH --job-name=2room-theirs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-41
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_theirs_%A_%a.log
# ====== TWOROOM: SPEC-ACCEPT AT LeWM'S OWN PUBLISHED PROTOCOL (Fig. 6) ======
# Bars + predictions frozen in docs/tworoom_paired_prereg.md ("THEIR-PROTOCOL
# BATTERY") BEFORE any cell here ran. Protocol = paper App. F.1/D verbatim
# (start random, goal = +25 steps same trajectory via goal_proprio, budget 50,
# H5 blocks, full-block replanning) + holdout hygiene (episodes >= 4000), n=64.
# Router hop threshold 0.163 = the gap probe's hop-S10 p50 (derived constant).
# ============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

GDM=gdm_tworoom_s10_gc_paired_e400v.pt
SPEC=(--subgoal specpaired --gdm-ckpt "$GDM" --gdm-steps 50 --accept-tau 0.098)
BOK=(--best-of-k 8 --bok-score goal)
RT=(--route-goal-hop 0.163)
THEIRS=(--mode short --start random --goal-offset 25 --eval-budget 50
        --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 6))
case $((ID / 6)) in
  0) NAME=flat_rh5;        ARGS=(--subgoal baseline --receding-horizon 5) ;;
  1) NAME=flat_rh2;        ARGS=(--subgoal baseline --receding-horizon 2) ;;
  2) NAME=oracle_s10_rh5;  ARGS=(--subgoal oracle --stride 10 --receding-horizon 5) ;;
  3) NAME=pair_rh2;        ARGS=("${SPEC[@]}" --receding-horizon 2) ;;
  4) NAME=pair_bok_rh2;    ARGS=("${SPEC[@]}" "${BOK[@]}" --receding-horizon 2) ;;
  5) NAME=pair_bok_rt_rh2; ARGS=("${SPEC[@]}" "${BOK[@]}" "${RT[@]}" --receding-horizon 2) ;;
  6) NAME=pair_bok_rt_rh5; ARGS=("${SPEC[@]}" "${BOK[@]}" "${RT[@]}" --receding-horizon 5) ;;
esac

LOG=logs/2room_theirs_${NAME}_seed${SEED}.log
echo "=== their-protocol arm=$NAME seed=$SEED ==="
python -m specaccept.envs.tworoom.eval \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  "${THEIRS[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "=== DONE $NAME seed=$SEED ==="
