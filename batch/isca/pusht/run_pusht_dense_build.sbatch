#!/usr/bin/env bash
#SBATCH --job-name=pusht-dense
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_dense_%j.log
# ISCA port of run_build_dense_isambard.sbatch: rebuild subgoals_dense_full.pt
# (stride-1, every frame of every successful@20deg episode, 18,685 eps) with
# the frozen pretrained encoder. Independent re-derivation of the PushT
# pipeline on new hardware -- the stride-S drafters train from this ONE file.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
python -m specaccept.envs.pusht.build_subgoals --source pretrained \
  --h5 "$H5" --out subgoals_dense_full.pt --stride 1 --device cuda --batch-size 256
