#!/usr/bin/env bash
#SBATCH --job-name=stride-seeds2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_seeds2_%A_%a.log
# Seeds 46-49 for every point on the SR-vs-stride curve -> 8 seeds total per
# stride, halving the headline claim's standard error (s10 96.4% at 5deg vs
# s25 86.3% deserves tight CIs).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s5 s10 s15 s25)
CKPTS=(gdm_stride5.pt gdm_stride10.pt gdm_stride15.pt gdm_stride25.pt)
RHS=(1 2 3 5)
SEEDS=(46 47 48 49)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; CKPT=${CKPTS[$C]}; RH=${RHS[$C]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/stride
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== arm=$NAME seed=$SEED horizon=receding=$RH n=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt ${CKPT} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/stride/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/stride_seeds2_${NAME}_seed${SEED}.log
