#!/usr/bin/env bash
#SBATCH --job-name=reacher-arms
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_arms_%A_%a.log
# Reacher closed loop: the PushT winner battery transplanted. 6 configs x
# seeds 42-45, n=128 held-out episodes (>=8000), LeWM reacher protocol
# (goal_offset 25, budget 50, qpos_match 0.05 rad), S = RH * action_block(5):
#   c0 baseline   flat LeWM, goal image      (RH 5)  -- the LeWM reference
#   c1 s25 gdm    every-step, native scale   (RH 5)  -- FF-JEPA-native config
#   c2 s10 gdm    every-step                 (RH 2)  -- PushT scale-law winner
#   c3 s5  gdm    every-step                 (RH 1)
#   c4 s10 spec   tau=0.20 k=8               (RH 2)  -- PushT headline system
#   c5 s5  spec   tau=0.20 k=8               (RH 1)  -- PushT 8-seed best cell
# Success criteria: (i) scale law reproduces (s10/s5 >= s25), (ii) spec-accept
# SR within noise of its every-step reference at call_ratio < 1 ([specaccept]
# telemetry line), (iii) gdm arms >= baseline.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=$(find /scratch/u6ko/hasoshu.u6ko/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
NAMES=(baseline s25gdm s10gdm s5gdm s10spec s5spec)
SUBGOALS=(baseline gdm gdm gdm specaccept specaccept)
CKPTS=(none gdm_reacher_s25.pt gdm_reacher_s10.pt gdm_reacher_s5.pt gdm_reacher_s10.pt gdm_reacher_s5.pt)
RHS=(5 5 2 1 2 1)
KS=(50 50 50 50 8 8)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; SG=${SUBGOALS[$C]}; CKPT=${CKPTS[$C]}; RH=${RHS[$C]}; K=${KS[$C]}
CKPT_ARG=""
if [ "$CKPT" != "none" ]; then CKPT_ARG="--gdm-ckpt $CKPT"; fi
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/reacher
echo "=== arm=$NAME subgoal=$SG ckpt=$CKPT horizon=receding=$RH k=$K seed=$SEED ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  export MUJOCO_GL=egl
  python -m ffjepa.eval_reacher --subgoal ${SG} ${CKPT_ARG} \
    --accept-tau 0.20 --gdm-steps ${K} \
    --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
    --num-eval 128 --seed ${SEED} --episode-min 8000 \
    --goal-offset 25 --eval-budget 50 \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/reacher/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/reacher_arms_${NAME}_seed${SEED}.log
