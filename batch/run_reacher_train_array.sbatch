#!/usr/bin/env bash
#SBATCH --job-name=reacher-train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_train_%A_%a.log
# Reacher drafters at S in {5,10,25}: the EXACT faithful recipe from
# run_train_stride_array.sbatch (batch 128, lr 5e-5, wd 1e-3, clip 1.0,
# warmup_cosine, 20 ep, EMA off, bf16 amp), with two Reacher-specific changes:
#   * GOAL-CONDITIONED (--goal-cond --goal-rule window --goal-gap 25): the
#     dataset is random-policy, so goal-free drafting is ill-posed (the TwoRoom
#     lesson). Hindsight window goal = the latent 25 frames ahead (= the eval
#     protocol's goal_offset), targets clamped at the goal -- the same clamp
#     semantics PushT gets free from demos ending at the target.
#   * HOLDOUT: train on episodes 0..7999 only (--mask custom); eval draws from
#     8000..9999 -- the H8 eval-contamination guard from the PushT audit.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
STRIDES=(5 10 25)
S=${STRIDES[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python - <<'PY'
import torch, numpy as np
blob = torch.load('subgoals_reacher_dense.pt', map_location='cpu', weights_only=False)
ep = blob['episode_idx'].numpy()
mask = ep < 8000
np.save('reacher_train_mask.npy', mask)
print(f'[mask] train episodes: {mask.sum()}/{len(mask)} (holdout = {int((~mask).sum())})')
PY
  echo '=== train gdm_reacher_s${S}.pt (faithful recipe, goal-cond window/gap25, eps<8000) ==='
  python -m ffjepa.train_gdm --subgoals subgoals_reacher_dense.pt --out gdm_reacher_s${S}.pt \
    --device cuda --mask custom --custom-mask-file reacher_train_mask.npy \
    --subgoal-step ${S} --goal-cond --goal-rule window --goal-gap 25 \
    --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
    --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
"
