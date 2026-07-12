#!/usr/bin/env bash
#SBATCH --job-name=reacher-frontier
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_frontierjob_%A_%a.log
# SR-vs-compute frontier for spec-accept at the winning cell (t100, 2t budget,
# s10 drafter, max-offset-100 population -- directly comparable to the main
# horizon table's t100 cells): tau {0.1,0.2,0.3,0.4} x k {4,8,16} x seeds
# {42,43}. Characterizes the efficiency claim quantitatively (NFE per replan
# = call_ratio * k vs every-step's 1.0 * 50). tau=0.1 approximates the
# near-every-step limit at cheap k.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
TAUS=(0.1 0.2 0.3 0.4)
KKS=(4 8 16)
SEEDS=(42 43)
T=$((SLURM_ARRAY_TASK_ID / 6))
K=$(((SLURM_ARRAY_TASK_ID / 2) % 3))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
TAU=${TAUS[$T]}; KK=${KKS[$K]}
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
EPFILE=reacher_horizon.ep.frontier.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_frontier_tau${TAU}_k${KK}_seed${SEED}.log
echo "=== frontier: s10spec tau=$TAU k=$KK seed=$SEED (t100, budget 200) ==="

source "$VENV/bin/activate"
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 100 --episode-min 8000 --seed 42
rm -f "$LOG"
python -m specaccept.envs.reacher.eval --subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt \
  --accept-tau ${TAU} --gdm-steps ${KK} \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset 100 --eval-budget 200 \
  --horizon 2 --receding-horizon 2 \
  --dump-traces runs/reacher/traces_frontier_tau${TAU}_k${KK}_seed${SEED}.pt \
  2>&1 | tee -a "$LOG"
