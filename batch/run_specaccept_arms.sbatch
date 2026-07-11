#!/usr/bin/env bash
#SBATCH --job-name=spec-arms
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/spec_arms_%A_%a.log
# Phase B closed loop: speculative acceptance with the REALITY verifier at the
# Stage 3/4 winning stride (S=10; 97.2/96.4 every-step reference from the sweep)
# plus an S=5 point. 5 configs x seeds 42-45:
#   c0 s10 tau=0.15 k=8   c1 s10 tau=0.20 k=8   c2 s10 tau=0.30 k=8
#   c3 s10 tau=0.20 k=50 (isolates few-step drafting from acceptance)
#   c4 s5  tau=0.20 k=8  (A2-calibrated tau at fine stride)
# Success criterion: SR within noise of the matching every-step arm at a
# measured call_ratio <= ~0.6 ([specaccept] telemetry line per run).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s10t15k8 s10t20k8 s10t30k8 s10t20k50 s5t20k8)
CKPTS=(gdm_stride10.pt gdm_stride10.pt gdm_stride10.pt gdm_stride10.pt gdm_stride5.pt)
TAUS=(0.15 0.20 0.30 0.20 0.20)
KS=(8 8 8 50 8)
RHS=(2 2 2 2 1)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; CKPT=${CKPTS[$C]}; TAU=${TAUS[$C]}; K=${KS[$C]}; RH=${RHS[$C]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/specaccept
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== arm=$NAME ckpt=$CKPT tau=$TAU k=$K horizon=receding=$RH seed=$SEED n=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.eval_ffjepa --subgoal specaccept --gdm-ckpt ${CKPT} \
    --accept-tau ${TAU} --gdm-steps ${K} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/specaccept/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/spec_arms_${NAME}_seed${SEED}.log
