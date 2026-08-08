#!/usr/bin/env bash
#SBATCH --job-name=sgc-curve
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-13%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/sgc_curve_%A_%a.log
# ========= P-EXEC-4 / P-EXEC-5: THE FULL EXECUTOR CURVE + DO-NO-HARM ========
# prereg/2026-08-07_composite_and_executor.md, amendment 2026-08-08, frozen
# before this submission. Completes the executor exhibit:
#   tasks 0-5   PushT t{25,50,75} x seeds{42,43}: SURVEYOR-Base w/ GC-IDM
#               executor vs GC-IDM their-spec (100.0/96.9/94.9). Bar: >= -2pp
#               each t; prediction: the gap is monotone nondecreasing in t.
#   tasks 6-13  Reacher t{25,50,100,150} x seeds{42,43}: the layer on GC-IDM's
#               home turf. Band: +/-2pp of plain GC-IDM per cell; -2..-5 =
#               measured decomposition tax; < -5 anywhere = the layer harms a
#               strong amortized policy, reported at full prominence.
# Their own Appendix G names this hybrid as future work; these cells measure
# the certified version of it. Reacher specgcidm wiring is new -> every
# reacher task runs an n=16 smoke first and gates on it (smokes never quoted).
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

if [ $ID -le 5 ]; then
  POFFS=(25 25 50 50 75 75)
  OFF=${POFFS[$ID]}; SEED=$((42 + ID % 2)); BUD=$((2 * OFF))
  EPFILE=pusht.episodes150s5as${OFF}.json
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== P-EXEC-4 pusht t=$OFF seed=$SEED n=$N ==="
  python -m specaccept.envs.pusht.eval --subgoal specgcidm \
    --gcidm-ckpt gcidm_pusht_h50.pt --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
    --accept-tau 0.20 --sg-steps 10 \
    --source pretrained --h5 "$PUSHT_H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget ${BUD} \
    --num-eval $N --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    2>&1 | tee logs/sgc_curve_pusht_t${OFF}_seed${SEED}.log
else
  REL=$((ID - 6))
  ROFFS=(25 25 50 50 100 100 150 150)
  OFF=${ROFFS[$REL]}; SEED=$((42 + REL % 2)); BUD=$((2 * OFF))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
  for f in gdm_reacher_s10.pt gcidm_reacher_h50.pt "$EPFILE"; do
    [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

  echo "=== reacher specgcidm SMOKE n=16 (plumbing gate; never quoted) ==="
  python -m specaccept.envs.reacher.eval --subgoal specgcidm \
    --gcidm-ckpt gcidm_reacher_h50.pt --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
    --accept-tau 0.20 --sg-steps 10 \
    --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
    --start random --goal-offset 25 --eval-budget 50 --num-eval 16 \
    --episode-min 8000 --seed 42 --cem-seed 42 \
    2>&1 | tee logs/sgc_curve_reacher_smoke_task${ID}.log
  grep -q "SR = " logs/sgc_curve_reacher_smoke_task${ID}.log || { echo "SMOKE FAILED"; exit 1; }

  echo "=== P-EXEC-5 reacher t=$OFF seed=$SEED ==="
  python -m specaccept.envs.reacher.eval --subgoal specgcidm \
    --gcidm-ckpt gcidm_reacher_h50.pt --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
    --accept-tau 0.20 --sg-steps 10 \
    --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
    --num-eval 128 --seed $SEED --cem-seed $SEED --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget ${BUD} \
    2>&1 | tee logs/sgc_curve_reacher_t${OFF}_seed${SEED}.log
fi
echo "=== sgc-curve task $ID DONE ==="
