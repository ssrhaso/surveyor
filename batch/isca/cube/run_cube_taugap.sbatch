#!/usr/bin/env bash
#SBATCH --job-name=cube-taugap
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeTG_%A_%a.log
# ================== TAU-FROM-GAP PRESCRIPTION TEST (pre-registered) ==================
# The gap statistic upgrades from EXPLANATORY to PRESCRIPTIVE if reading tau off
# the gap replaces the auxiliary mechanism it diagnosed. Cube's diagnosis
# (gap_cube.json): gap = [0.219, 0.353], tau=0.20 sits just BELOW it -> slight
# over-rejection -> why cube needed the arrival gate. This arm: gc15, NO gate,
# tau = 0.286 (gap midpoint), certified protocol verbatim, 8 seeds.
# BANKED REFERENCES (same protocol, seeds 42-45, cubeG battery):
#   gc15_nogate @ tau0.20 = 76.76 mean (77.34/75.00/75.00/79.69)
#   gc15_gate   @ tau0.20 = 79.30 mean   flat = 70.9   oracle = 82.4
# BARS (frozen now): PASS = tau-in-gap recovers >= half the gate margin without
# the gate (8-seed mean >= 78.0); mechanistic secondary = reject count drops vs
# the banked nogate logs. FAIL = mean <= 76.76 ==> cube's gate effect is NOT
# over-rejection; amend the instrument's cube diagnosis honestly.
# =====================================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "=== cube-taugap gc15 NOGATE tau=0.286 seed=$SEED (certified: 600 samples, offset150 budget300) ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt gdm_cube_s15_gc.pt \
  --accept-tau 0.286 --gdm-steps 8 --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeTG_taugap_seed${SEED}.log
