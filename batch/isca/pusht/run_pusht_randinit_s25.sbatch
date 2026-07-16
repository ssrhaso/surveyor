#!/usr/bin/env bash
#SBATCH --job-name=pusht-ri-s25
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_ris25job_%A_%a.log
# Last cell of the "FF-JEPA configuration (our run)" headline row: random-init
# at THEIR exact config (S=25, every-step, k=50, RH=5) on the certified
# pipeline. Box-era repro was 96.5 (n=256, 1 seed); this makes it 4 seeds
# certified so the entire row carries no box-era footnote.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
SEED=$((42 + SLURM_ARRAY_TASK_ID))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
echo "=== pusht random-init FF-JEPA config (s25, k=50): seed=$SEED ==="
python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-steps 50 \
  --gdm-ckpt gdm_stride25.pt \
  --source pretrained --h5 "$H5" --device cuda \
  --mode random_init --score block --angles 20 5 --num-eval 256 \
  --seed $SEED --cem-seed $SEED \
  --horizon 5 --receding-horizon 5 \
  2>&1 | tee logs/pusht_randinit_s25gdm_seed${SEED}.log
