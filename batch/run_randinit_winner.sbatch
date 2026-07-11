#!/usr/bin/env bash
#SBATCH --job-name=randinit-winner
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/randinit_winner_%A_%a.log
# Random-init n=256 for the stride winners -- the paper's weakest cell (its
# random-init: 82.42%; our gdm_faithful/stride-25 repro: 96.48/78.12 at
# 20/5deg, seed 42). If the fine-stride precision story transfers, the 5deg
# number is where it shows. Loop geometry scaled to the stride as in Stage 3.
#   task 0: s10 seed 42   task 1: s10 seed 43   task 2: s5 seed 42
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s10 s10 s5)
CKPTS=(gdm_stride10.pt gdm_stride10.pt gdm_stride5.pt)
RHS=(2 2 1)
SEEDS=(42 43 42)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}; CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
RH=${RHS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm
echo "=== random_init arm=$NAME seed=$SEED horizon=receding=$RH n=256 ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt ${CKPT} \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --device cuda \
    --mode random_init --score block --angles 20 5 --num-eval 256 \
    --seed ${SEED} --cem-seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH}
" 2>&1 | tee logs/randinit_${NAME}_seed${SEED}.log
