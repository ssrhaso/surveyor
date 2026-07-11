#!/usr/bin/env bash
#SBATCH --job-name=commit2-s10
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/commit2_s10_%A_%a.log
# Stage 4's pre-registered commit-2 point at the best stride (S=10): consume
# the raw drafted m+1 THEN m+2 blindly (no verification) before re-anchoring --
# the one commit-depth point unmeasured at fine spacing. Uses the validated
# DSpark raw/fixed path (--no-refine --commit fixed --commit-k 2). Contrast
# with the specaccept arms (verified consumption at the same effective depth):
# the gap between blind-2 and verified-2 IS the value of the reality verifier.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
SEEDS=(42 43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
echo "=== commit2 s10 seed=$SEED n=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal dspark --gdm-ckpt gdm_stride10.pt \
    --no-refine --commit fixed --commit-k 2 \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val.episodes.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 5 --num-eval $NEV --seed ${SEED} \
    --horizon 2 --receding-horizon 2
" 2>&1 | tee logs/commit2_s10_seed${SEED}.log
