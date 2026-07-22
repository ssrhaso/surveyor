#!/usr/bin/env bash
#SBATCH --job-name=tworoom-s1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/tworoom_s1_%j.log
# ============ TWOROOM UN-PARKED (user, 2026-07-21): STAGE 1 =================
# The 7/5 kill was a LATENT-FIDELITY judgment; V-JEPA 2 stage-4 (2026-07-20)
# showed that metric does not track planning value (lerp best-rel-err/worst
# plans; v1-gdm worst-rel-err/best far plans). Retry as the paper's
# INTERVENTION experiment: goal-free fails (scope condition, already
# measured) -> supplying the goal restores applicability -> does the
# goal-conditioned CERTIFIED method work?
# Stages (idempotent): A download+schema gate; B ANCHOR (cube lesson):
# baseline flat SR at t=25/75, cross-room population, episodes >= 4000,
# n=64, seeds 42-43; C build S=10 gc pairs from episodes < 4000 (head);
# D train gc drafter (goal = episode's final subgoal). The unified arm is
# NOT run here; its bars get pre-registered against B's anchor first.
# tau=0.20 frozen (transfer default; re-derive only on failure, paper rule).
# ==============================================================================
set -euo pipefail
DATA=/lustre/home/ha676/data/tworoom
H5DIR=$DATA
cd /lustre/home/ha676/le-wm
mkdir -p logs
source .venv/bin/activate
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy

echo "=== stage A: download + schema gate ==="
if ! ls "$H5DIR"/*.h5 "$H5DIR"/*.hdf5 2>/dev/null | grep -q .; then
  mkdir -p "$DATA"; cd "$DATA"
  curl -sL https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst \
    | zstd -d --stdout | tar -xf -
  find "$DATA" -maxdepth 3 -exec ls -lhd {} \;
  cd /lustre/home/ha676/le-wm
fi
H5=$(find "$H5DIR" \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
echo "H5=$H5"
# Published tar ships pos_agent/pos_target; the drivers + swm callables expect
# state/goal_state (TwoRoomEnv._set_state = agent pos, _set_goal_state =
# target pos; verified identical semantics). Alias in-place, idempotent.
python - "$H5" <<'EOF'
import sys, h5py
with h5py.File(sys.argv[1], "a") as f:
    for dst, src in (("state", "pos_agent"), ("goal_state", "pos_target")):
        if dst not in f:
            f.create_dataset(dst, data=f[src][:])
            print(f"aliased {src} -> {dst}")
    f.visititems(lambda n, o: print(f"  {n}  {getattr(o,'shape','')}") if hasattr(o, "shape") else None)
    print("episodes:", len(f["ep_len"]))
EOF

echo "=== stage B: ANCHOR (baseline flat, cross-room, holdout eps>=4000) ==="
for OFF in 25 75; do
  BUD=$((2 * OFF))
  for SEED in 42 43; do
    LOG=logs/tworoom_anchor_flat_t${OFF}_seed${SEED}.log
    if [ -f "$LOG" ] && grep -q "SR =" "$LOG"; then echo "skip $LOG"; continue; fi
    python -m specaccept.envs.tworoom.eval --subgoal baseline \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode $([ "$OFF" -le 25 ] && echo short || echo long) \
      --goal-offset $OFF --eval-budget $BUD --start final \
      --eval-filter success --require-cross-room --episode-min 4000 \
      --num-eval 64 --seed $SEED --cem-seed $SEED \
      2>&1 | tee "$LOG"
  done
done

echo "=== stage C: build S=10 gc pairs (train episodes < 4000, head) ==="
SG=subgoals_tworoom_s10.pt
if [ ! -f "$SG" ]; then
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 "$H5" --out "$SG" --source pretrained \
    --encoder-id quentinll/lewm-tworooms --device cuda \
    --stride 10 --max-episodes 4000 --sample-mode head \
    2>&1 | tee logs/tworoom_build_s10.log
fi

echo "=== stage D: train gc drafter (S=10, goal=final subgoal) ==="
CKPT=gdm_tworoom_s10_gc.pt
if [ ! -f "$CKPT" ]; then
  python -m specaccept.train_drafter \
    --subgoals "$SG" --out "$CKPT" --goal-cond --goal-rule final \
    --mask 5 --device cuda --seed 42 \
    2>&1 | tee logs/tworoom_train_gc.log
fi
echo "=== TWOROOM STAGE 1 DONE ==="
