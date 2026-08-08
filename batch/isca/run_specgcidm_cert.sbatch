#!/usr/bin/env bash
#SBATCH --job-name=sgc-cert
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/sgc_cert_%A_%a.log
# ============ P-EXEC-6: THE CERTIFIED EXECUTOR ARM (c* over GC-IDM) =========
# prereg/2026-08-07_composite_and_executor.md amendment 2, frozen before this
# submission. P-EXEC-5's falsifier fired at short reacher (drafting tax,
# executor-independent); this arm adds the certificate at both scopes: one
# flat-CEM c* probe at the first boundary routes to plain GC-IDM (window 2x5,
# tau unchanged), and the tau arrival gate retires drafting envs. One CEM
# solve per episode; execution fully amortised.
#   tasks 0-7  reacher t{25,50,100,150} x seeds{42,43}
#              bars: t25/t50 >= GC-IDM - 2pp (recovery); t100 >= 90.4,
#              t150 >= 94.3 (retention)
#   task 8     pusht t150 seed 42: range safety, within +/-1.5pp of 96.09
# Falsifiers (either shelves the arm, composite carries): reacher short
# < GC-IDM - 5pp; pusht t150 < 94.0. Smokes gate plumbing, never quoted.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ID=$SLURM_ARRAY_TASK_ID
PUSHT_H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
REACHER_H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

if [ $ID -le 7 ]; then
  ROFFS=(25 25 50 50 100 100 150 150)
  OFF=${ROFFS[$ID]}; SEED=$((42 + ID % 2)); BUD=$((2 * OFF))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json

  echo "=== cert SMOKE n=16 (never quoted) ==="
  python -m specaccept.envs.reacher.eval --subgoal specgcidm --cstar-route \
    --gcidm-ckpt gcidm_reacher_h50.pt --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
    --accept-tau 0.20 --sg-steps 10 \
    --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
    --start random --goal-offset 25 --eval-budget 50 --num-eval 16 \
    --episode-min 8000 --seed 42 --cem-seed 42 \
    2>&1 | tee logs/sgc_cert_reacher_smoke_task${ID}.log
  grep -q "SR = " logs/sgc_cert_reacher_smoke_task${ID}.log || { echo "SMOKE FAILED"; exit 1; }

  echo "=== P-EXEC-6 reacher t=$OFF seed=$SEED ==="
  python -m specaccept.envs.reacher.eval --subgoal specgcidm --cstar-route \
    --gcidm-ckpt gcidm_reacher_h50.pt --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
    --accept-tau 0.20 --sg-steps 10 \
    --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
    --num-eval 128 --seed $SEED --cem-seed $SEED --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget ${BUD} \
    2>&1 | tee logs/sgc_cert_reacher_t${OFF}_seed${SEED}.log
else
  EPFILE=pusht.episodes150s5.json
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== P-EXEC-6c pusht t=150 range safety n=$N ==="
  python -m specaccept.envs.pusht.eval --subgoal specgcidm --cstar-route \
    --gcidm-ckpt gcidm_pusht_h50.pt --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
    --accept-tau 0.20 --sg-steps 10 \
    --source pretrained --h5 "$PUSHT_H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
    --num-eval $N --score block --angles 20 --seed 42 --cem-seed 42 \
    2>&1 | tee logs/sgc_cert_pusht_t150_seed42.log
fi
echo "=== sgc-cert task $ID DONE ==="
