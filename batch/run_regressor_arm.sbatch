#!/usr/bin/env bash
#SBATCH --job-name=regressor-arm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/regressor_arm_%A_%a.log
# The deferred diffusion-vs-regression closed-loop arm (last open SR question):
# deterministic z_cond->m+1 regressor (offline-dominant JointMLP) as the subgoal
# source, re-anchoring every replan, at S=10 (winner) and S=25 (original
# deferred comparison). Task 0 of each stride trains the regressor first
# (minutes); the rest wait on a done-marker. Spread thesis predicts the
# regressor LOSES to diffusion (zero conditional spread = regression-to-mean
# target); references: s10 gdm 97.18/96.37, s25 gdm 91.13/86.29.
#   tasks 0-3: S=10 seeds 42-45   tasks 4-7: S=25 seeds 42-45
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
STRIDES=(10 25)
RHS=(2 5)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
S=${STRIDES[$C]}; RH=${RHS[$C]}
CKPT=regressor_stride${S}.pt
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/regressor
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")

# first task per stride trains; the others wait for the marker (train ~ minutes)
if [ $((SLURM_ARRAY_TASK_ID % 4)) -eq 0 ]; then
  apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
    source $VENV/bin/activate
    python -m specaccept.train_regressor --subgoals subgoals_dense_full.pt \
      --subgoal-step ${S} --out ${CKPT} --device cuda
  "
  touch ${CKPT}.done
else
  for i in $(seq 1 240); do [ -f ${CKPT}.done ] && break; sleep 15; done
  [ -f ${CKPT}.done ] || { echo "TRAIN MARKER TIMEOUT"; exit 1; }
fi

echo "=== regressor arm S=$S seed=$SEED horizon=receding=$RH n=$NEV ==="
apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal regressor --regressor-ckpt ${CKPT} \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/regressor/traces_s${S}_seed${SEED}.pt
" 2>&1 | tee logs/regressor_s${S}_seed${SEED}.log
