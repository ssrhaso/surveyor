#!/usr/bin/env bash
#SBATCH --job-name=reacher-hz-base
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_hzbase_%A_%a.log
# The missing baseline arm of the horizon sweep: flat LeWM (goal image, no
# drafter) on the SAME fixed episode population as run_reacher_horizon_sweep,
# offsets {25,50,75,100} x seeds {42,43}, both budget regimes. Without this
# the drafter arms' ~90% at t100 (2t regime) is uninterpretable; baseline
# beat every drafter arm by ~24-30pp on the native task, so the only
# pre-registered place the method can win must be measured against baseline
# too. t25 cell re-run here (rather than reusing the arms battery) so the
# population matches the sweep exactly (episodes-file, not episode-min draw).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFSETS=(25 50 75 100)
SEEDS=(42 43)
O=$((SLURM_ARRAY_TASK_ID / 2))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
EPFILE=reacher_horizon.ep.base.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_horizon_baseline_t${OFF}_seed${SEED}.log
echo "=== arm=baseline offset=$OFF seed=$SEED (budgets: 2t=$((2*OFF)), vlwm=50) ==="

source "$VENV/bin/activate"
# deterministic (seed 42): identical population to the drafter sweep tasks
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 100 --episode-min 8000 --seed 42
rm -f "$LOG"
echo '--- FF-JEPA regime: budget 2t ---'
python -m specaccept.envs.reacher.eval --subgoal baseline \
  --accept-tau 0.20 --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon 5 --receding-horizon 5 \
  --dump-traces runs/reacher/traces_hz_baseline_t${OFF}_2t_seed${SEED}.pt \
  2>&1 | tee -a "$LOG"
if [ ${OFF} -ne 25 ]; then
  echo '--- VLWM regime: budget 50 fixed ---'
  python -m specaccept.envs.reacher.eval --subgoal baseline \
    --accept-tau 0.20 --gdm-steps 50 \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget 50 \
    --horizon 5 --receding-horizon 5 \
    --dump-traces runs/reacher/traces_hz_baseline_t${OFF}_b50_seed${SEED}.pt \
    2>&1 | tee -a "$LOG"
fi
