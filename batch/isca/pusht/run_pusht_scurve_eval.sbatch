#!/usr/bin/env bash
#SBATCH --job-name=pusht-scurve-ev
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_scurve_ev_%A_%a.log
# Evals for the completed PushT long-horizon S curve: S {15,5} x {t150 fixed
# pop (budget 300), native short (success5, budget 50)} x seeds 42-45.
# Loop matched: S=15 -> RH3, S=5 -> RH1. Every-step k=50 (SR-reference config,
# comparable to the existing s10/s25 spine cells).
#   idx = s*8 + proto*4 + seed_idx ; s: 0->15 1->5 ; proto: 0->t150 1->short
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
if [ $((ID / 8)) -eq 0 ]; then S=15; RH=3; else S=5; RH=1; fi
PROTO=$(((ID / 4) % 2))
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $PROTO -eq 0 ]; then
  echo "=== s-curve eval: S=$S t150 seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride${S}.pt \
    --gdm-steps 50 --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon $RH --receding-horizon $RH \
    2>&1 | tee logs/pusht_s${S}_t150_seed${SEED}.log
else
  echo "=== s-curve eval: S=$S native short seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride${S}.pt \
    --gdm-steps 50 --source pretrained --h5 "$H5" --device cuda \
    --mode short --eval-filter success5 --num-eval 256 \
    --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon $RH --receding-horizon $RH \
    2>&1 | tee logs/pusht_s${S}_short_seed${SEED}.log
fi
