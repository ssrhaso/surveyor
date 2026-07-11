#!/usr/bin/env bash
#SBATCH --job-name=reacher-horizon
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_horizon_%A_%a.log
# ISCA port of batch/run_reacher_horizon_sweep.sbatch (Isambard). Fixed-
# population horizon sweep: one episode population (same 128 pairs, holdout
# >= 8000) at every offset; per task BOTH regimes (FF-JEPA budget=2t, VLWM
# budget=50 fixed, skipped at t=25 where they coincide). Arms x offsets x
# seeds: {s25gdm, s10gdm, s10spec} x {25,50,75,100} x {42,43}.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(s25gdm s10gdm s10spec)
SUBGOALS=(gdm gdm specaccept)
CKPTS=(gdm_reacher_s25.pt gdm_reacher_s10.pt gdm_reacher_s10.pt)
RHS=(5 2 2)
KS=(50 50 8)
OFFSETS=(25 50 75 100)
SEEDS=(42 43)
A=$((SLURM_ARRAY_TASK_ID / 8))
O=$(((SLURM_ARRAY_TASK_ID / 2) % 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; CKPT=${CKPTS[$A]}; RH=${RHS[$A]}; K=${KS[$A]}
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
EPFILE=reacher_horizon.ep.t${SLURM_ARRAY_TASK_ID}.json
echo "=== arm=$NAME offset=$OFF seed=$SEED (budgets: 2t=$((2*OFF)), vlwm=50) ==="

source "$VENV/bin/activate"
# deterministic (seed 42): every task regenerates identical content, no races
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 100 --episode-min 8000 --seed 42
echo '--- FF-JEPA regime: budget 2t ---'
python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} \
  --accept-tau 0.20 --gdm-steps ${K} \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_2t_seed${SEED}.pt \
  2>&1 | tee logs/reacher_horizon_${NAME}_t${OFF}_2t_seed${SEED}.log
if [ ${OFF} -ne 25 ]; then
  echo '--- VLWM regime: budget 50 fixed ---'
  python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} \
    --accept-tau 0.20 --gdm-steps ${K} \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget 50 \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_b50_seed${SEED}.pt \
    2>&1 | tee logs/reacher_horizon_${NAME}_t${OFF}_b50_seed${SEED}.log
fi
