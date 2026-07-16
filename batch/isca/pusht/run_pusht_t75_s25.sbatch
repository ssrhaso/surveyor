#!/usr/bin/env bash
#SBATCH --job-name=pusht-t75-s25
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_t75s25job_%A_%a.log
# Retires the last dagger in tab:headline2: FF-JEPA config (S=25, every-step,
# k=50, RH=5) at the t=75 paper protocol on the certified pipeline (box-era
# repro was 90.0, published 91.8). 4 seeds, n=256 each.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
SEED=$((42 + SLURM_ARRAY_TASK_ID))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
echo "=== pusht t75 paper-protocol, FF-JEPA config (s25 k=50): seed=$SEED ==="
python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-steps 50 \
  --gdm-ckpt gdm_stride25.pt \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --goal-offset 75 --eval-budget 150 --eval-filter success5 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 5 --receding-horizon 5 \
  2>&1 | tee logs/pusht_t75_s25gdm_seed${SEED}.log
