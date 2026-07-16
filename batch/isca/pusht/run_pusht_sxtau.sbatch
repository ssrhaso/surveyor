#!/usr/bin/env bash
#SBATCH --job-name=pusht-sxtau
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_sxtaujob_%A_%a.log
# S x tau interaction grid (closes the sec:calib-scale separability TBD, and
# documents the check(iv) prediction that the tau band's upper edge widens with
# S). Fills the idle nodes during the cube-training window; finishes well before
# the cube battery needs them. S{5,10,25} x tau{0.1,0.2,0.3,0.4}, k=8, t100
# anchor, matched loop (RH=S/5), seed 42, n=256.
#   S = {5,10,25}[idx//4]; TAU = {.1,.2,.3,.4}[idx%4]
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
SS=(5 10 25); TAUS=(0.10 0.20 0.30 0.40); RHS=(1 2 5)
SI=$((ID / 4)); S=${SS[$SI]}; RH=${RHS[$SI]}; TAU=${TAUS[$((ID % 4))]}
CKPT=gdm_stride${S}.pt
echo "=== pusht SxTau: S=$S tau=$TAU k=8 RH=$RH t100 seed42 ==="
python -m specaccept.envs.pusht.eval --subgoal specaccept --accept-tau $TAU \
  --gdm-ckpt $CKPT --gdm-steps 8 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 5 --seed 42 --cem-seed 42 \
  --horizon $RH --receding-horizon $RH \
  2>&1 | tee logs/pusht_sxtau_S${S}_tau${TAU}_t100_seed42.log
