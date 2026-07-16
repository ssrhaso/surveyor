#!/usr/bin/env bash
#SBATCH --job-name=pusht-randinit
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_randinitjob_%A_%a.log
# Headline-table gap fill (2026-07-13): the random-init column was box-era
# (footnote b) and had no spec-accept cell (dash). Re-run on the certified
# ISCA pipeline: every-step at the box-era-comparable config (k=50) to retire
# footnote b against the 100.0, and spec-accept at the DERIVED config
# (tau=.20, k=3) to fill the dash. Paper III-A protocol: fully random init,
# goal = final frame of a random successful episode, per-episode goals
# (sample_random_init_goals), n=256/seed, seeds 42-45.
#   0-3 every-step gdm k=50, seeds 42-45     4-7 spec tau=.2 k=3, seeds 42-45
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $ID -le 3 ]; then NAME=gdm; ARM="--subgoal gdm --gdm-steps 50"; else NAME=speck3; ARM="--subgoal specaccept --accept-tau 0.20 --gdm-steps 3"; fi
echo "=== pusht random-init: $NAME seed=$SEED (n=256, per-episode goals) ==="
python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt \
  --source pretrained --h5 "$H5" --device cuda \
  --mode random_init --score block --angles 20 5 --num-eval 256 \
  --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_randinit_${NAME}_seed${SEED}.log
