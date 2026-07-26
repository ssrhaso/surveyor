#!/usr/bin/env bash
#SBATCH --job-name=dwm-specsmoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_specsmoke_%j.log
# Spec-serving smoke: 2 evals through SpecMPCPlanner (draft -> verify -> serve
# latent goal -> unmodified CEM). Gates the battery. Chained afterok drafter.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python patch_cem_latentgoal.py
ARM=spec TAU=0.121 N_EVALS=2 MAX_ITER=6 .venv/bin/python run_plan_pusht_spec.py plan_outputs/isca_specsmoke
echo "=== DWM SPEC SMOKE OK ==="
