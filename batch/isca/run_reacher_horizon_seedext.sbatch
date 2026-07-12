#!/usr/bin/env bash
#SBATCH --job-name=reacher-hz-ext
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_hzext_%A_%a.log
# Seed extension for the ONE positive Reacher cell: the t100 (2t-budget)
# crossover where s10gdm (91.8) beats baseline (82.8) at n=2 seeds. Adds
# seeds 44,45 at t in {75,100} for {baseline, s10gdm, s10spec}, 2t regime
# only (the starved regime is already decisive, baseline +15-25pp there).
# Same fixed seed-42 episode population as every other horizon job; the
# aggregator merges these into the existing cells (n=2 -> n=4).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(baseline s10gdm s10spec)
SUBGOALS=(baseline gdm specaccept)
CKPTS=(none gdm_reacher_s10.pt gdm_reacher_s10.pt)
RHS=(5 2 2)
KS=(50 50 8)
OFFSETS=(75 100)
SEEDS=(44 45)
A=$((SLURM_ARRAY_TASK_ID / 4))
O=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; CKPT=${CKPTS[$A]}; RH=${RHS[$A]}; K=${KS[$A]}
OFF=${OFFSETS[$O]}
CKPT_ARG=""
if [ "$CKPT" != "none" ]; then CKPT_ARG="--gdm-ckpt $CKPT"; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
EPFILE=reacher_horizon.ep.ext.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_horizon_${NAME}_t${OFF}_seed${SEED}.log
echo "=== arm=$NAME offset=$OFF seed=$SEED (2t budget=$((2*OFF))) ==="

source "$VENV/bin/activate"
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 100 --episode-min 8000 --seed 42
rm -f "$LOG"
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 --gdm-steps ${K} \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_2t_seed${SEED}.pt \
  2>&1 | tee -a "$LOG"
