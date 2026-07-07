#!/usr/bin/env bash
#SBATCH --job-name=stride-ab
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_ab_%A_%a.log
# Stage 3 (stride-spacing round): closed-loop A/B/C, 3 configs x 4 CEM seeds,
# all on the same held-out n=62 episodes (dspark_val.episodes.json), goal_offset
# 150, eval-budget 300, block criterion 20/5deg. Anchor: gdm stride-25 commit-1
# = 91.9%/83.1% (seeds 42-45 mean).
#   s5  : stride-5 drafter, horizon=receding=1  -> 5-step targets, re-anchor
#         every 5 (the fine-control package: stride+cadence+depth move together
#         by construction, DIRECTION_stride_spacing.md 2)
#   s25 : stride-25 drafter, horizon=receding=5 -> the baseline package, run
#         through the SAME rebuilt pipeline (re-derivation of the 91.9 anchor)
#   cad : stride-25 drafter, horizon=5, receding=1 -> CADENCE-ISOLATION control:
#         re-anchor every 5 env-steps (like s5) but 25-step targets and full
#         25-step CEM rollouts (like s25). Separates "re-anchor more often"
#         from "closer target / shallower search". Well-posed (rollout length
#         ~ target distance), unlike holding horizon=5 in the s5 arm (25-step
#         terminal cost toward a 5-step target = the degenerate mismatch the
#         direction doc rules out).
# Read-off: s5 vs s25 = does fine control granularity pay. cad vs s25 = does
# cadence ALONE pay at fixed stride/depth. s5 vs cad = does the finer TARGET
# add anything beyond cadence.
# Diffusion calls/episode (GDMSubgoalSource re-drafts every replan): s5=60,
# cad=60, s25=12.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
NAMES=(s5 s25 cad)
CKPTS=(gdm_stride5.pt gdm_stride25.pt gdm_stride25.pt)
HORS=(1 5 5)
RECS=(1 5 1)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; CKPT=${CKPTS[$C]}; H=${HORS[$C]}; RH=${RECS[$C]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/stride
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== arm=$NAME ckpt=$CKPT horizon=$H receding=$RH seed=$SEED num_eval=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.eval_ffjepa --subgoal gdm --gdm-ckpt ${CKPT} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${H} --receding-horizon ${RH} \
    --dump-traces runs/stride/traces_${NAME}_seed${SEED}.pt
" 2>&1 | tee logs/stride_ab_${NAME}_seed${SEED}.log
