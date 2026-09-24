#!/usr/bin/env bash
#SBATCH --job-name=surveyor
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.log
# One array task per manifest row: the array index is the row's idx column. Submit from the repository root, with the
# site's account and partition flags added:
#   mkdir -p logs
#   sbatch --array=0-711 --export=ALL,MANIFEST=reproduce/battery_A.tsv,DATA=/path/to/datasets reproduce/run_manifest.sbatch
# The paper's wall-clock columns were measured with one row per job on one GPU.
set -uo pipefail
: "${MANIFEST:?pass --export=ALL,MANIFEST=reproduce/battery_<X>.tsv}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array job}"
exec bash reproduce/run_manifest.sh "$MANIFEST" "$SLURM_ARRAY_TASK_ID"
