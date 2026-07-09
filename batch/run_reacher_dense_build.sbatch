#!/usr/bin/env bash
#SBATCH --job-name=reacher-dense
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_dense_%j.log
# Reacher Stage 1 prerequisite: dense (stride-1) latent build over ALL 10,000
# random-policy episodes with the frozen lewm-reacher encoder -- the reacher
# analog of subgoals_dense_full.pt. Stride-S drafters then train from this one
# file via train_gdm --subgoal-step S (the PushT stride-sweep scheme).
# Run after run_reacher_download_inspect.sbatch (needs reacher.h5 + encoder_reacher).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=$(find /scratch/u6ko/hasoshu.u6ko/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /home/u6ko/hasoshu.u6ko/le-wm
echo "=== dense build from $H5 ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.build_subgoals_reacher --source local --local-dir encoder_reacher \
    --h5 $H5 --out subgoals_reacher_dense.pt --stride 1 --device cuda --batch-size 256
"
