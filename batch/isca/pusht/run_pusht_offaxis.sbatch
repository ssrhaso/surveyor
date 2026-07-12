#!/usr/bin/env bash
#SBATCH --job-name=pusht-offaxis
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_offaxisjob_%A_%a.log
# tau x k separability spot-check on PushT (off-axis corners): the one-knob-
# at-a-time sweep design rests on no-interaction, measured on Reacher (full
# 12-cell cross) but only assumed on PushT (L-shaped grid). Two off-axis
# cells x 2 seeds: if the k=4 ordering and tau slope match the on-axis grids,
# separability transfers and the design is justified on both envs.
#   0-1 tau=0.30 k=4 seeds 42-43   2-3 tau=0.10 k=4 seeds 42-43
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
TAU=$([ $((ID / 2)) -eq 0 ] && echo 0.30 || echo 0.10)
SEED=$((42 + ID % 2))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
echo "=== off-axis: tau=$TAU k=4 seed=$SEED (t100) ==="
python -m specaccept.envs.pusht.eval --subgoal specaccept --gdm-ckpt gdm_stride10.pt \
  --accept-tau $TAU --gdm-steps 4 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_offaxis_tau${TAU}_k4_seed${SEED}.log
