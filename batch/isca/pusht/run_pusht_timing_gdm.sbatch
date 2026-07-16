#!/usr/bin/env bash
#SBATCH --job-name=pusht-timing-g
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_timingG_%A_%a.log
# Completes the measured latency ledger: the FF-JEPA GDM reference arms,
# instrumented (the earlier timing cells covered spec/unified; flat's cost is
# arithmetic from the measured CEM solve x replan count since the bare
# baseline bypasses the instrumented policy).
#   0: GDM every-step k=50, S=10 loop (RH=2)  -- our-substrate every-step ref
#   1: GDM every-step k=50, S=25 loop (RH=5)  -- the FF-JEPA native config
# PushT t=100, s5 spine population, seed 42, n=256, budget 200.
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  NAME=gdm50_s10; CKPT=gdm_stride10.pt; RH=2
else
  NAME=gdm50_s25; CKPT=gdm_stride25.pt; RH=5
fi
echo "=== pusht timing-gdm arm=$NAME t=100 seed=42 (time-instrument) ==="
python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-steps 50 \
  --gdm-ckpt $CKPT \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150s5as100.json \
  --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 --seed 42 --cem-seed 42 \
  --horizon ${RH} --receding-horizon ${RH} \
  --time-instrument \
  2>&1 | tee logs/pusht_timingG_${NAME}.log
