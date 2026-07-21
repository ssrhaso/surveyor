#!/usr/bin/env bash
#SBATCH --job-name=pusht-horizon
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-39
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_hz_%A_%a.log
# PushT fixed-population horizon battery on ISCA, WITH the baseline arm the
# original (box-era) design lacked; the Reacher lesson applied back to
# PushT. One 256-episode population (pusht.episodes150*.json), budgets 2t.
# Audit anchors: baseline collapses long (21.48 @t75, 1.17 @t150 old pipeline);
# gdm holds ~90.
#   ARM(4) x OFFSET(5: 25,50,75,100,150) x SEED(2): idx = a*10 + o*2 + s
#   arms: baseline(RH5) s25gdm(RH5) s10gdm(RH2) s10spec(RH2 tau.2 k8)
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
ARMS=(baseline s25gdm s10gdm s10spec)
OFFSETS=(25 50 75 100 150)
SEEDS=(42 43)
A=$((ID / 10)); O=$(((ID / 2) % 5)); SEED=${SEEDS[$((ID % 2))]}
NAME=${ARMS[$A]}; OFF=${OFFSETS[$O]}
case $NAME in
  baseline) ARM="--subgoal baseline --horizon 5 --receding-horizon 5" ;;
  s25gdm)   ARM="--subgoal gdm --gdm-ckpt gdm_stride25.pt --horizon 5 --receding-horizon 5" ;;
  s10gdm)   ARM="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
  s10spec)  ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2" ;;
esac
if [ $OFF -eq 150 ]; then EPFILE=pusht.episodes150.json; else EPFILE=pusht.episodes150as${OFF}.json; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
echo "=== pusht-horizon arm=$NAME offset=$OFF budget=$((2*OFF)) seed=$SEED ==="
python -m specaccept.envs.pusht.eval $ARM \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset $OFF --eval-budget $((2*OFF)) \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  2>&1 | tee logs/pusht_hz_${NAME}_t${OFF}_seed${SEED}.log
