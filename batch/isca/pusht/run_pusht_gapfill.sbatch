#!/usr/bin/env bash
#SBATCH --job-name=pusht-gapfill
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_gapfilljob_%A_%a.log
# Two remaining PushT gap-fills (2026-07-13 eve):
# (B) t=75 PAPER PROTOCOL on the certified pipeline; the last box-era number
#     cited in the main text (headline footnote d: "ours at t75: 97.9" was
#     box-era k=8). Paper-matched sampling (success5 filter, offset 75,
#     budget 150, n=256). every-step k=50 = comparable to published 91.8;
#     spec at the DERIVED config.
# (C) S x tau interaction mini-grid; floor-theory corollary: the tau band's
#     UPPER edge should scale with the real-divergence scale, i.e. coarser S
#     tolerates looser tau (old offline calibration hinted tau=0.4 for S=25).
#     S=25 drafter (RH=5 matched loop), k=8 (in-band for that drafter; k=3 was
#     derived on the S=10 drafter and does not transfer untested), t100 anchor.
#   0-3   t75 every-step k=50, seeds 42-45
#   4-7   t75 spec tau=.2 k=3, seeds 42-45
#   8-11  S25 spec tau=.2 k=8, seeds 42-45  (t100 unfiltered anchor)
#   12-15 S25 spec tau=.4 k=8, seeds 42-45
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $ID -le 7 ]; then
  if [ $ID -le 3 ]; then NAME=t75_gdm; ARM="--subgoal gdm --gdm-steps 50"; else NAME=t75_speck3; ARM="--subgoal specaccept --accept-tau 0.20 --gdm-steps 3"; fi
  echo "=== gap-fill B: paper-protocol t75 $NAME seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --goal-offset 75 --eval-budget 150 --eval-filter success5 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_${NAME}_seed${SEED}.log
else
  if [ $ID -le 11 ]; then TAU=0.20; NAME=s25spec_tau02; else TAU=0.40; NAME=s25spec_tau04; fi
  echo "=== gap-fill C: S=25 spec tau=$TAU k=8 t100 seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal specaccept --accept-tau $TAU \
    --gdm-ckpt gdm_stride25.pt --gdm-steps 8 \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 5 --receding-horizon 5 \
    2>&1 | tee logs/pusht_${NAME}_t100_seed${SEED}.log
fi
