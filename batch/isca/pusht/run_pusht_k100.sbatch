#!/usr/bin/env bash
#SBATCH --job-name=pusht-k100
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_k100job_%A_%a.log
# Matched k-freeze anchor (audit follow-up): Reacher's k grid sits at t=100,
# PushT's at t=150; a cross-env k disagreement would be confounded with
# horizon. This runs the full PushT k grid AT t=100 (same anchor as Reacher):
# {every-step, spec tau=.2} x k {4,8} x seeds 42-45, fixed population
# (episodes150as100 reindex), budget 200, RH2.
#   idx = arm*8 + kidx*4 + seed_idx ; arms: 0=every-step 1=spec ; k: 0->4 1->8
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
KS=(4 8)
K=${KS[$(((ID / 4) % 2))]}
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $((ID / 8)) -eq 0 ]; then NAME=gdmk${K}; ARM="--subgoal gdm"; else NAME=speck${K}; ARM="--subgoal specaccept --accept-tau 0.20"; fi
echo "=== pusht k-anchor t100: $NAME seed=$SEED ==="
python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt --gdm-steps $K \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_${NAME}_t100_seed${SEED}.log
