#!/usr/bin/env bash
#SBATCH --job-name=vlwm-regime
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/vlwm_regime_%A_%a.log
# Our three horizon-sweep arms under the BeyondNext/VLWM regime: evaluation
# budget FIXED at 50 env steps regardless of goal offset (their App. B), same
# fixed 256-episode population as job 5579798. t=25/budget-50 already exists
# (it IS the t=25 cells of 5579798), so this covers t in {50,75,100}. At
# t=100/budget-50 success requires beating the expert's pace 2x; expect a
# real drop; how much we retain vs LeWM's 8% / VLWM's 12% is the honest
# cross-paper claim (curve shapes still the primary comparison).
#   task = CONFIG(3) x OFFSET(3) x SEED(2): idx = c*6 + o*2 + s
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
CONFIGS=(s25 s10 spec)
OFFSETS=(50 75 100)
SEEDS=(42 43)
C=$((SLURM_ARRAY_TASK_ID / 6))
O=$(((SLURM_ARRAY_TASK_ID / 2) % 3))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
CFG=${CONFIGS[$C]}; OFF=${OFFSETS[$O]}
EPFILE=subset_longeval.episodes150as${OFF}.json
cd /home/u6ko/hasoshu.u6ko/le-wm

case "$CFG" in
  s25)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_faithful.pt --horizon 5 --receding-horizon 5" ;;
  s10)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
  spec) ARM_ARGS="--subgoal specaccept --accept-tau 0.20 --gdm-steps 8 --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
esac
echo "=== vlwm-regime cfg=$CFG offset=$OFF budget=50 seed=$SEED epfile=$EPFILE ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval $ARM_ARGS \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file $EPFILE --device cuda \
    --mode long --goal-offset $OFF --eval-budget 50 \
    --score block --angles 20 5 --num-eval 256 --seed ${SEED} --cem-seed ${SEED}
" 2>&1 | tee logs/vlwm_regime_${CFG}_t${OFF}_seed${SEED}.log
