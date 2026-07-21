#!/usr/bin/env bash
#SBATCH --job-name=pusht-blind2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_blind2job_%A_%a.log
# THE missing matched-compute cell (2026-07-13 review): blind commit-2 (no
# verifier: --subgoal dspark --no-refine --commit fixed --commit-k 2, raw GDM
# block, call ratio 0.50 by construction) at MATCHED k, on the same fixed
# unfiltered anchors as the spec-accept cells. Table 5's old comparison ran
# blind at k=50 vs spec at k=8; never matched. If blind-2 ties spec here,
# PushT has no evidence the verifier does anything in-distribution (only the
# k-cliff remains); if it loses, verification has direct in-distribution
# evidence. The k=2 rows are the cliff variant: spec degrades to every-step
# there (~55); does blind coast BELOW that?
#   0-3  blind2 k=4 t100 seeds 42-45     4-7  blind2 k=8 t100 seeds 42-45
#   8-11 blind2 k=4 t150 seeds 42-45    12-15 blind2 k=8 t150 seeds 42-45
#  16-17 blind2 k=2 t100 seeds 42-43  (cliff)
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if   [ $ID -le 3 ];  then K=4; OFF=100; SEED=$((42 + ID));
elif [ $ID -le 7 ];  then K=8; OFF=100; SEED=$((42 + ID - 4));
elif [ $ID -le 11 ]; then K=4; OFF=150; SEED=$((42 + ID - 8));
elif [ $ID -le 15 ]; then K=8; OFF=150; SEED=$((42 + ID - 12));
else                      K=2; OFF=100; SEED=$((42 + ID - 16)); fi
if [ $OFF -eq 100 ]; then EPFILE=pusht.episodes150as100.json; BUDGET=200;
else                      EPFILE=pusht.episodes150.json;      BUDGET=300; fi
echo "=== pusht blind commit-2: k=$K t$OFF seed=$SEED ==="
python -m specaccept.envs.pusht.eval \
  --subgoal dspark --no-refine --commit fixed --commit-k 2 \
  --gdm-ckpt gdm_stride10.pt --gdm-steps $K \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset $OFF --eval-budget $BUDGET \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_blind2k${K}_t${OFF}_seed${SEED}.log
