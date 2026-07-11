#!/usr/bin/env bash
#SBATCH --job-name=tworoom-train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/tworoom_train_%A_%a.log
# TwoRoom drafters at S in {5,10,15,25}: the faithful recipe (batch 128,
# lr 5e-5, wd 1e-3, clip 1.0, warmup_cosine, 20 ep, EMA off, bf16 amp) on the
# existing crossroom dense file, GOAL-CONDITIONED with the FINAL rule (expert
# demos end at their per-episode target, so the episode-end goal is exact --
# no hindsight window needed, unlike Reacher's random-policy data). Goal-cond
# is required here: TwoRoom's target is per-episode and unrendered, so
# goal-free drafting is structurally ill-posed (the parked-phase finding; this
# array is the scale-curve version of gdm_tworoom_goalcond.pt).
# HOLDOUT: train on episodes < 4000 of source file 0 only (--mask custom);
# eval draws episodes >= 4000 -- the H8 contamination guard.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
DENSE=subgoals_tworoom_dense_crossroom.pt
STRIDES=(5 10 15 25)
S=${STRIDES[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python - <<'PY'
import torch, numpy as np
blob = torch.load('$DENSE', map_location='cpu', weights_only=False)
ep = blob['episode_idx'].numpy()
src = blob.get('source_file')
src = src.numpy() if src is not None else np.zeros(len(ep), dtype=int)
mask = (ep < 4000) & (src == 0)
np.save('tworoom_train_mask.npy', mask)
print(f'[mask] train episodes: {mask.sum()}/{len(mask)} '
      f'(holdout >=4000 of source 0: {int(((ep >= 4000) & (src == 0)).sum())}); '
      f\"eval h5 = {blob.get('source_files', ['?'])[0]}\")
PY
  echo '=== train gdm_tworoom_s${S}.pt (faithful recipe, goal-cond final rule, eps<4000) ==='
  python -m specaccept.train_drafter --subgoals $DENSE --out gdm_tworoom_s${S}.pt \
    --device cuda --mask custom --custom-mask-file tworoom_train_mask.npy \
    --subgoal-step ${S} --goal-cond \
    --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
    --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
"
