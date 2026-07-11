#!/usr/bin/env bash
#SBATCH --job-name=spec-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/spec_smoke_%j.log
# Smoke gate for SpecAcceptSubgoalSource (new code): 4 episodes at the winning
# stride (S=10). The full overnight arm array is submitted with
# --dependency=afterok on this job, so a code bug costs 10 minutes, not a night.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
cd /home/u6ko/hasoshu.u6ko/le-wm
python3 -c "
import json
p = json.load(open('dspark_val.episodes.json'))
p['episodes'] = p['episodes'][:4]
json.dump(p, open('dspark_val_smoke.json', 'w'))
print('smoke episodes:', p['episodes'])
"
apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal specaccept --gdm-ckpt gdm_stride10.pt \
    --accept-tau 0.20 --gdm-steps 8 \
    --source local --local-dir $ENC --h5 subset_longeval.h5 \
    --episodes-file dspark_val_smoke.json --device cuda \
    --mode long --goal-offset 150 --eval-budget 300 \
    --score block --angles 20 --num-eval 4 --seed 42 \
    --horizon 2 --receding-horizon 2
"
grep -q "specaccept" logs/spec_smoke_${SLURM_JOB_ID}.log && echo SMOKE_OK
