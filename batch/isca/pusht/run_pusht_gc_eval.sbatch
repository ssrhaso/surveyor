#!/usr/bin/env bash
#SBATCH --job-name=pusht-gc
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_gcjob_%A_%a.log
# The last 2x2 ablation cell: PushT x goal-conditioned drafter
# (gdm_stride10_gc, final-rule hindsight goals). References: goal-free s10
# short 99.4 / t150 (unfiltered pop) 59.6. Question: does conditioning help,
# hurt, or nothing when the data is already goal-directed (expert demos)?
# Reacher answered the converse (wander data NEEDS conditioning: 10% vs 60%).
#   0-3 short protocol seeds 42-45     4-7 t150 unfiltered pop seeds 42-45
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $ID -le 3 ]; then
  echo "=== gc-ablation: short seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride10_gc.pt \
    --gdm-steps 50 --source pretrained --h5 "$H5" --device cuda \
    --mode short --eval-filter success5 --num-eval 256 \
    --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_gc_short_seed${SEED}.log
else
  echo "=== gc-ablation: t150 seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride10_gc.pt \
    --gdm-steps 50 --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_gc_t150_seed${SEED}.log
fi
