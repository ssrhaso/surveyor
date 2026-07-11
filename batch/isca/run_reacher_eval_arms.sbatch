#!/usr/bin/env bash
#SBATCH --job-name=reacher-arms
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-35
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_arms_%A_%a.log
# ISCA port of batch/run_reacher_eval_arms.sbatch (Isambard). The PushT winner
# battery transplanted: 9 configs x seeds 42-45, n=128 held-out episodes
# (>=8000), LeWM reacher protocol (goal_offset 25, budget 50, qpos_match 0.05
# rad). Success criteria: (i) scale law reproduces, (ii) spec-accept SR within
# noise of every-step at call_ratio < 1, (iii) gdm arms >= baseline.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
NAMES=(baseline s25gdm s15gdm s10gdm s5gdm s25spec s15spec s10spec s5spec)
SUBGOALS=(baseline gdm gdm gdm gdm specaccept specaccept specaccept specaccept)
CKPTS=(none gdm_reacher_s25.pt gdm_reacher_s15.pt gdm_reacher_s10.pt gdm_reacher_s5.pt gdm_reacher_s25.pt gdm_reacher_s15.pt gdm_reacher_s10.pt gdm_reacher_s5.pt)
RHS=(5 5 3 2 1 5 3 2 1)
KS=(50 50 50 50 50 8 8 8 8)
SEEDS=(42 43 44 45)
C=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${NAMES[$C]}; SG=${SUBGOALS[$C]}; CKPT=${CKPTS[$C]}; RH=${RHS[$C]}; K=${KS[$C]}
CKPT_ARG=""
if [ "$CKPT" != "none" ]; then CKPT_ARG="--gdm-ckpt $CKPT"; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
echo "=== arm=$NAME subgoal=$SG ckpt=$CKPT horizon=receding=$RH k=$K seed=$SEED ==="

source "$VENV/bin/activate"
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 --gdm-steps ${K} \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episode-min 8000 \
  --goal-offset 25 --eval-budget 50 \
  --horizon ${RH} --receding-horizon ${RH} \
  --dump-traces runs/reacher/traces_${NAME}_seed${SEED}.pt \
  2>&1 | tee logs/reacher_arms_${NAME}_seed${SEED}.log
