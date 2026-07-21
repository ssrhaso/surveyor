#!/usr/bin/env bash
#SBATCH --job-name=pusht-rh
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-27
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_rhjob_%A_%a.log
# PushT strong-baseline replan-rate sweep (the Reacher lesson applied to
# PushT, which never had it): flat LeWM baseline at RH in {1,2} across the
# short protocol and every horizon offset. RH=5 rows come from the
# replication/horizon batteries. Every paper table then anchors to the
# strongest flat configuration per cell. Reacher precedent: RH=2 flat hit
# 95.7 at t25 (vs protocol 84.0) but collapsed to 66.6 at t100; if PushT
# reproduces that brittleness, the "subgoals resolve the replan-rate
# trade-off" finding is cross-env.
#   0-7   short protocol (success5, n=256, budget 50): RH1 x seeds 42-45,
#         RH2 x seeds 42-45
#   8-27  horizon (fixed population, budget 2t): RH{1,2} x t{25,50,75,100,150}
#         x seeds{42,43}
# No drafter needed anywhere; runnable before tonight's training.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"

BASE="--subgoal baseline --source pretrained --h5 $H5 --device cuda \
  --num-eval 256 --score block --angles 20 5"

if [ $ID -le 7 ]; then
  RH=$((1 + ID / 4))
  SEED=$((42 + ID % 4))
  echo "=== pusht rh-sweep: short protocol RH=$RH seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $BASE \
    --mode short --eval-filter success5 --seed $SEED --cem-seed $SEED \
    --horizon $RH --receding-horizon $RH \
    2>&1 | tee logs/pusht_rh${RH}_short_seed${SEED}.log
else
  REL=$((ID - 8))
  RH=$((1 + REL / 10))
  REST=$((REL % 10))
  OFFSETS=(25 50 75 100 150)
  OFF=${OFFSETS[$((REST / 2))]}
  SEED=$((42 + REST % 2))
  if [ $OFF -eq 150 ]; then EPFILE=pusht.episodes150.json; else EPFILE=pusht.episodes150as${OFF}.json; fi
  echo "=== pusht rh-sweep: horizon RH=$RH t=$OFF budget=$((2*OFF)) seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $BASE \
    --mode long --episodes-file $EPFILE --goal-offset $OFF --eval-budget $((2*OFF)) \
    --seed $SEED --cem-seed $SEED \
    --horizon $RH --receding-horizon $RH \
    2>&1 | tee logs/pusht_rh${RH}_t${OFF}_seed${SEED}.log
fi
