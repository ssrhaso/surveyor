#!/usr/bin/env bash
#SBATCH --job-name=comp-arms
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/comp1111_arms_%A_%a.log
# ============= P-COMP-1: INSTRUMENT-ROUTED COMPOSITE, FRESH CELLS ============
# prereg/2026-08-07_composite_and_executor.md section A; frozen assignments
# PushT->SURVEYOR, Reacher->GC-IDM (their H_max=T spec), Cube->GC-IDM.
#
# Verdict reduction, stated openly: "composite >= every fixed arm" is decided
# by the MAX fixed arm per cell, and with GC-IDM in the arm set the banked max
# is the composite's own arm in 9 of 10 cells (the exception, pusht t=25, is
# covered by the GC-IDM reference tasks). Runner-up checks run where banked
# margins were thin (reacher t25/t50 flat RH2/RH5). Cube's runner-up trails
# by 13pp banked and is cited rather than re-run tonight; extension pooling
# happens Saturday if any fresh margin lands inside 3pp.
#
#  0-9   pusht SURVEYOR (unified c*): t{25,50,75,100,150} x seeds{42,43}
#  10-14 pusht GC-IDM their-spec (H=2t ckpts): 5 offsets, deterministic
#  15-18 reacher GC-IDM their-spec: t{25,50,100,150}
#  19-22 reacher flat runner-up: RH{2,5} x t{25,50}, seeds 42+43 in-task
#  23    cube GC-IDM their-spec: fresh seeds 52-59 in-task
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

PUSHT_H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
REACHER_H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
OFFS=(25 50 75 100 150)

if [ $ID -le 9 ]; then
  OFF=${OFFS[$((ID / 2))]}; SEED=$((42 + ID % 2)); BUD=$((2 * OFF))
  EPFILE=pusht_c1111.episodes${OFF}.json
  echo "=== comp1111 pusht SURVEYOR t=$OFF seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal unified \
    --gdm-ckpt gdm_stride10.pt --gdm-steps 3 --accept-tau 0.20 \
    --horizon 2 --receding-horizon 2 \
    --source pretrained --h5 "$PUSHT_H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget ${BUD} \
    --num-eval 256 --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
    2>&1 | tee logs/comp1111_pusht_unified_t${OFF}_seed${SEED}.log
elif [ $ID -le 14 ]; then
  OFF=${OFFS[$((ID - 10))]}; BUD=$((2 * OFF))
  EPFILE=pusht_c1111.episodes${OFF}.json
  CKPT=gcidm_pusht_h${BUD}.pt
  echo "=== comp1111 pusht GC-IDM their-spec t=$OFF H_max=$BUD ==="
  python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
    --source pretrained --h5 "$PUSHT_H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget ${BUD} \
    --num-eval 256 --score block --angles 20 5 --seed 42 --cem-seed 42 \
    2>&1 | tee logs/comp1111_pusht_gcidm_t${OFF}.log
elif [ $ID -le 18 ]; then
  ROFFS=(25 50 100 150)
  OFF=${ROFFS[$((ID - 15))]}; BUD=$((2 * OFF))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_c1111.ep${MAXOFF}.s1111.json
  CKPT=gcidm_reacher_h${BUD}.pt
  echo "=== comp1111 reacher GC-IDM their-spec t=$OFF H_max=$BUD ==="
  python -m specaccept.envs.reacher.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
    --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
    --num-eval 128 --seed 42 --cem-seed 42 --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget ${BUD} \
    2>&1 | tee logs/comp1111_reacher_gcidm_t${OFF}.log
elif [ $ID -le 22 ]; then
  REL=$((ID - 19)); RH=2; if [ $((REL / 2)) -eq 1 ]; then RH=5; fi
  ROFFS=(25 50); OFF=${ROFFS[$((REL % 2))]}; BUD=$((2 * OFF))
  EPFILE=reacher_c1111.ep100.s1111.json
  for SEED in 42 43; do
    echo "=== comp1111 reacher flat RH=$RH t=$OFF seed=$SEED ==="
    python -m specaccept.envs.reacher.eval --subgoal baseline --receding-horizon $RH \
      --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
      --num-eval 128 --seed ${SEED} --cem-seed ${SEED} --episodes-file $EPFILE \
      --goal-offset ${OFF} --eval-budget ${BUD} \
      2>&1 | tee logs/comp1111_reacher_flat_rh${RH}_t${OFF}_seed${SEED}.log
  done
else
  for S in 52 53 54 55 56 57 58 59; do
    echo "=== comp1111 cube GC-IDM their-spec seed=$S (fresh population) ==="
    python -m specaccept.envs.cube.eval --subgoal gcidm --gcidm-ckpt gcidm_cube_h300.pt \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 128 --seed $S --cem-seed $S --episode-min 8000 \
      2>&1 | tee logs/comp1111_cube_gcidm_seed${S}.log
  done
fi
echo "=== comp1111 task $ID DONE ==="
