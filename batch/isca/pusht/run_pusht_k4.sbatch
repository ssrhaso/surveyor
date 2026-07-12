#!/usr/bin/env bash
#SBATCH --job-name=pusht-k4
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_k4job_%A_%a.log
# Stage-0 k-freeze cells (the gap in the k grid): k=4 on PushT, both arms,
# at the controlled t150 cell. Joint k decision = whichever of {4,8} is
# defensible on BOTH envs (Reacher: k=4 dominates; PushT: unknown until now).
#   0-3 every-step k=4, seeds 42-45      4-7 spec tau=.2 k=4, seeds 42-45
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $ID -le 3 ]; then NAME=gdmk4; ARM="--subgoal gdm"; else NAME=speck4; ARM="--subgoal specaccept --accept-tau 0.20"; fi
echo "=== pusht k=4: $NAME seed=$SEED (t150) ==="
python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt --gdm-steps 4 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_${NAME}_t150_seed${SEED}.log
