#!/usr/bin/env bash
#SBATCH --job-name=specgcidm
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/specgcidm_%A.log
# ========== P-EXEC-1: GC-IDM AS THE EXECUTOR INSIDE SURVEYOR ================
# prereg/2026-08-07_composite_and_executor.md section B, frozen BEFORE this
# submission. Mechanism bet: GC-IDM degrades with goal DISTANCE (its Table 2);
# the drafter manufactures NEAR goals (S=10 hops, mid-distribution for the
# H_max=50 checkpoint, h=10/50). Accept rule is executor-agnostic; tau=0.20,
# k=3, S=10 all carried, no CEM anywhere in the arm.
#
# P-EXEC-1 (frozen): PushT t=150 fixed pop (n=256), budget 300, seeds 42/43:
#   SR(20deg) >= 90.0. AMBIGUOUS [75,90). FALSIFIER < 75 -> negative result,
#   reported at full prominence.
#   References: SURVEYOR-Base 98.1 | GC-IDM best-swept 86.33 / their-spec
#   60.55 | flat 3.9.
# P-EXEC-2: wall-clock/episode <= 20% of the CEM-executor spec arm.
# P-EXEC-3: call ratio in [0.35, 0.80].
# Smoke (n=16) gates plumbing only; smoke numbers are never quoted.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
GDM=gdm_stride10.pt
EXEC=gcidm_pusht_h50.pt
EXEC2=gcidm_pusht_h200.pt
for f in "$GDM" "$EXEC"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

run_cell () {  # $1=episodes-file $2=goal-offset $3=budget $4=num-eval $5=seed $6=exec-ckpt $7=logname
  python -m specaccept.envs.pusht.eval --subgoal specgcidm \
    --gcidm-ckpt "$6" --gdm-ckpt "$GDM" --gdm-steps 3 \
    --accept-tau 0.20 --sg-steps 10 \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file "$1" --goal-offset "$2" --eval-budget "$3" \
    --num-eval "$4" --score block --angles 20 5 --seed "$5" --cem-seed "$5" \
    2>&1 | tee "logs/$7"
}

EP150=pusht.episodes150s5.json
N150=$(python -c "import json;print(len(json.load(open('$EP150'))['episodes']))")

echo "=== SMOKE n=16 t=150 (plumbing gate; never quoted; self-sampled population" \
     "because --episodes-file always loads ALL its pairs) ==="
python -m specaccept.envs.pusht.eval --subgoal specgcidm \
  --gcidm-ckpt "$EXEC" --gdm-ckpt "$GDM" --gdm-steps 3 \
  --accept-tau 0.20 --sg-steps 10 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --goal-offset 150 --eval-budget 300 \
  --num-eval 16 --score block --angles 20 --seed 42 --cem-seed 42 \
  2>&1 | tee logs/specgcidm_smoke_t150.log
grep -q "\[RESULT\]" logs/specgcidm_smoke_t150.log || { echo "SMOKE FAILED - no RESULT line"; exit 1; }
echo "=== SMOKE PASSED, running frozen cells ==="

for SEED in 42 43; do
  echo "=== P-EXEC-1 t=150 n=$N150 seed=$SEED ==="
  run_cell "$EP150" 150 300 "$N150" "$SEED" "$EXEC" specgcidm_pusht_t150_seed${SEED}.log
done

EP100=pusht.episodes150s5as100.json
if [ -f "$EP100" ]; then
  N100=$(python -c "import json;print(len(json.load(open('$EP100'))['episodes']))")
  for SEED in 42 43; do
    echo "=== secondary t=100 n=$N100 seed=$SEED ==="
    run_cell "$EP100" 100 200 "$N100" "$SEED" "$EXEC" specgcidm_pusht_t100_seed${SEED}.log
  done
else
  echo "SKIP t=100 (no $EP100)"
fi

if [ -f "$EXEC2" ]; then
  echo "=== declared secondary executor H_max=200, t=150 seed=42 ==="
  run_cell "$EP150" 150 300 "$N150" 42 "$EXEC2" specgcidm_pusht_t150_h200_seed42.log
else
  echo "SKIP h200 secondary (no $EXEC2)"
fi
echo "=== ALL DONE ==="
