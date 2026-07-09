#!/usr/bin/env bash
#SBATCH --job-name=tworoom-arms
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-35
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/tworoom_arms_%A_%a.log
# TwoRoom closed loop: the scale-curve + spec-accept battery (writeup Sec.
# TwoRoom: SR vs subgoal scale under goal-conditioned planning). 9 configs x
# seeds 42-45, n=128 held-out cross-room episodes (>=4000, success-filtered),
# short protocol (goal_offset 25, budget 50, native dist<16px criterion),
# S = RH * action_block(5):
#   c0 baseline  flat LeWM, goal image      (RH 5)
#   c1-c4 every-step gdm at S=25/15/10/5    (RH 5/3/2/1) -- the scale curve
#   c5-c8 spec-accept tau=0.20 k=8 at same  (RH 5/3/2/1) -- efficiency arms
# The eval h5 is resolved from the dense file's source_files[0] (the same
# collection the drafters trained on; episodes >= 4000 are held out).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
DENSE=subgoals_tworoom_dense_crossroom.pt
NAMES=(baseline s25gdm s15gdm s10gdm s5gdm s25spec s15spec s10spec s5spec)
SUBGOALS=(baseline gdm gdm gdm gdm specaccept specaccept specaccept specaccept)
CKPTS=(none gdm_tworoom_s25.pt gdm_tworoom_s15.pt gdm_tworoom_s10.pt gdm_tworoom_s5.pt gdm_tworoom_s25.pt gdm_tworoom_s15.pt gdm_tworoom_s10.pt gdm_tworoom_s5.pt)
RHS=(5 5 3 2 1 5 3 2 1)
KS=(50 50 50 50 50 8 8 8 8)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; SG=${SUBGOALS[$C]}; CKPT=${CKPTS[$C]}; RH=${RHS[$C]}; K=${KS[$C]}
CKPT_ARG=""
if [ "$CKPT" != "none" ]; then CKPT_ARG="--gdm-ckpt $CKPT"; fi
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/tworoom
echo "=== arm=$NAME subgoal=$SG ckpt=$CKPT horizon=receding=$RH k=$K seed=$SEED ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  export SDL_VIDEODRIVER=dummy
  EVAL_H5=\$(python -c \"import torch; print(torch.load('$DENSE', map_location='cpu', weights_only=False)['source_files'][0])\")
  echo \"[arm] eval h5 = \$EVAL_H5\"
  python -m ffjepa.eval_tworoom --subgoal ${SG} ${CKPT_ARG} \
    --accept-tau 0.20 --gdm-steps ${K} \
    --source local --local-dir encoder_tworoom --h5 \$EVAL_H5 --device cuda \
    --num-eval 128 --seed ${SEED} --mode short \
    --eval-filter success --require-cross-room --episode-min 4000 \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/tworoom/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/tworoom_arms_${NAME}_seed${SEED}.log
