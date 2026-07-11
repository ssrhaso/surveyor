#!/usr/bin/env bash
#SBATCH --job-name=stride-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_sweep_%A_%a.log
# Stage 4 (stride-spacing round): complete the SR-vs-stride curve. Stage 3 gave
# S=5 (92.3/91.9) > S=25 (91.1/86.3) with the cadence-isolation control ruling
# out cadence-alone (78.6/77.8). This adds the interior points S in {10,15}
# (drafters already trained by the Stage 1 array), identical protocol:
# horizon = receding_horizon = S/5, same n=62 episodes, seeds 42-45, traces on.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s10 s15)
CKPTS=(gdm_stride10.pt gdm_stride15.pt)
RHS=(2 3)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; CKPT=${CKPTS[$C]}; RH=${RHS[$C]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/stride
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== arm=$NAME ckpt=$CKPT horizon=receding=$RH seed=$SEED num_eval=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt ${CKPT} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/stride/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/stride_sweep_${NAME}_seed${SEED}.log
