#!/usr/bin/env bash
#SBATCH --job-name=spec-seeds2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/spec_seeds2_%A_%a.log
# Seeds 46-49 for the three spec-accept headline-table arms -> 8 seeds per arm
# (pooled with the 42-45 runs), matching the stride curve's 8-seed power:
#   c0 A: s10 spec-accept tau=0.20 k=8  (spec-arms c1, s10t20k8, 5563730)
#   c1 B: s5  spec-accept tau=0.20 k=8  (spec-arms c4, s5t20k8,  5563730)
#   c2 C: s10 every-step gdm reference  (stride-sweep s10; default gdm-steps=50)
# Eval flags copied byte-identically from run_specaccept_arms.sbatch /
# run_stride_sweep.sbatch; ONLY the seed differs. Trace/log paths for arm C are
# NEW names on purpose: stride-seeds2 (5563731) already wrote
# runs/stride/traces_s10_seed4[6-9].pt + logs, which must not be clobbered.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s10t20k8 s5t20k8 s10every)
SUBS=(specaccept specaccept gdm)
CKPTS=(gdm_stride10.pt gdm_stride5.pt gdm_stride10.pt)
TAUS=(0.20 0.20 -)
KS=(8 8 -)
RHS=(2 1 2)
SEEDS=(46 47 48 49)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; SUB=${SUBS[$C]}; CKPT=${CKPTS[$C]}; TAU=${TAUS[$C]}; K=${KS[$C]}; RH=${RHS[$C]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/specaccept
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== arm=$NAME subgoal=$SUB ckpt=$CKPT tau=$TAU k=$K horizon=receding=$RH seed=$SEED n=$NEV ==="

if [ "$SUB" = specaccept ]; then
apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal specaccept --gdm-ckpt ${CKPT} \
    --accept-tau ${TAU} --gdm-steps ${K} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/specaccept/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/spec_arms_${NAME}_seed${SEED}.log
else
apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt ${CKPT} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/specaccept/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/spec_arms_${NAME}_seed${SEED}.log
fi
