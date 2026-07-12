#!/usr/bin/env bash
#SBATCH --job-name=pusht-repl
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_repl_%A_%a.log
# PushT short-protocol replication battery on ISCA (paper protocol: mode
# short, success5 population, n=256, 20deg/5deg scoring). Audit anchors this
# must reproduce within noise (pipeline certification):
#   baseline 93.36 / oracle 93.36 / s25 gdm 95.31 (20deg)
# plus the winning-config rows (s10 gdm, s10 spec) at 4 seeds each.
#   0-3   baseline  RH5, seeds 42-45
#   4-5   oracle    RH5, seeds 42-43
#   6-9   s25 gdm   RH5, seeds 42-45   (gdm_stride25 = gdm_faithful-equiv)
#   10-13 s10 gdm   RH2, seeds 42-45
#   14-17 s10 spec  RH2 tau=0.20 k=8, seeds 42-45
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"

case $ID in
  0|1|2|3)   NAME=baseline; SEED=$((42+ID));   ARM="--subgoal baseline --horizon 5 --receding-horizon 5" ;;
  4|5)       NAME=oracle;   SEED=$((42+ID-4)); ARM="--subgoal oracle --horizon 5 --receding-horizon 5" ;;
  6|7|8|9)   NAME=s25gdm;   SEED=$((42+ID-6)); ARM="--subgoal gdm --gdm-ckpt gdm_stride25.pt --horizon 5 --receding-horizon 5" ;;
  10|11|12|13) NAME=s10gdm; SEED=$((42+ID-10)); ARM="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
  14|15|16|17) NAME=s10spec; SEED=$((42+ID-14)); ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2" ;;
esac
echo "=== replication arm=$NAME seed=$SEED (short, success5, n=256) ==="
python -m specaccept.envs.pusht.eval $ARM \
  --source pretrained --h5 "$H5" --device cuda \
  --mode short --eval-filter success5 --num-eval 256 \
  --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  2>&1 | tee logs/pusht_repl_${NAME}_seed${SEED}.log
