#!/usr/bin/env bash
#SBATCH --job-name=specaccept-stride
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_train_%A_%a.log
# Stage 1 (stride-spacing round): retrain the GDM drafter at stride S in
# {5,10,15,25} -- the EXACT faithful recipe from run_recipe.sh (batch 128,
# lr 5e-5, wd 1e-3, clip 1.0, warmup_cosine, 20 ep, EMA off, bf16 amp, mask 5),
# only --subgoal-step changes (dense file_stride=1 => horizon = S frames).
# S=25 is the re-derivation sanity check: its diag curve must reproduce
# gdm_faithful.pt (Probe A/B rel_err ~0.16), proving the rebuilt dense file +
# retrain pipeline is faithful before any cross-stride comparison is read.
# Stage 2 offline gate (diag_gdm at matching --subgoal-step: rel_err vs
# noop_err degeneracy check) runs in-job right after training.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=/scratch/u6ko/hasoshu.u6ko/data/pusht/pusht_expert_train.h5
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
STRIDES=(5 10 15 25)
S=${STRIDES[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  echo '=== train gdm_stride${S}.pt (faithful recipe, subgoal-step ${S}) ==='
  python -m specaccept.train_drafter --subgoals subgoals_dense_full.pt --out gdm_stride${S}.pt \
    --device cuda --mask 5 --subgoal-step ${S} \
    --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
    --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
  echo '=== Stage 2 offline gate: diag at subgoal-step ${S} ==='
  python -m specaccept.diag_gdm --gdm-ckpt gdm_stride${S}.pt --subgoals subgoals_dense_full.pt \
    --subgoal-step ${S} --stride ${S} --h5 $H5 --device cuda --source local --local-dir $ENC
"
