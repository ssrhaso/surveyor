#!/usr/bin/env bash
# Reacher generalization chain for a bare GPU box (no SLURM, no apptainer) --
# local equivalent of submit_reacher_chain.sh + the run_reacher_*.sbatch files.
# Run from ~/le-wm after `source .venv/bin/activate` (batch/setup_local_env.sh):
#   bash batch/run_reacher_local.sh all
#   bash batch/run_reacher_local.sh download   # or: smoke | dense | train | arms | horizon
#
# Parallelism across MIG slices: set REACHER_GPU_DEVICES to a space-separated
# list of MIG UUIDs (from `nvidia-smi -L`) to fan the train/arms/horizon loops
# out across a dynamic worker pool (batch/dispatch_pool.sh) instead of running
# sequentially. Example (ofs-v03, 7 slices):
#   export REACHER_GPU_DEVICES="MIG-7b9f2944-027d-5890-a58a-09498dd3460f MIG-df629108-7b55-5f68-b2e7-984e62cd1b39 MIG-d8c24702-a6e6-57b6-bc9b-2d4d05c2568b MIG-4664404a-cf41-5dc4-a5ea-4bbbaa629971 MIG-b5cecf95-26f8-5318-b5a3-6195272b4eab MIG-59de5b88-d4e8-5159-bfcf-1d6eb0533555 MIG-548012f7-55c8-53cd-9af5-e55dddd140c4"
# Unset (default): falls back to plain sequential --device cuda, single slice.
# Whole stack is small (DiT hidden=512/depth=11, latent dim 192, CEM
# num_samples=300); a 10GB slice has plenty of headroom per job; the win
# from parallelism is wall-clock, not memory pressure.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-all}
DATA_DIR=${REACHER_DATA_DIR:-data/reacher}
export REACHER_DATA_DIR="$DATA_DIR"
# MIG slices are compute-only (no graphics APIs); EGL may fail there. If the
# smoke stage dies with an EGL/display error, rerun with MUJOCO_GL=osmesa
# (CPU software rendering; slower but MIG-safe).
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
mkdir -p logs runs/reacher "$DATA_DIR"

H5() { find "$DATA_DIR" \( -name '*.h5' -o -name '*.hdf5' \) 2>/dev/null | head -1; }

# Hard prerequisite checks: a missing dep or dataset must STOP the chain,
# not cascade tracebacks into the next stage (which is what happens when
# failures hide behind `| tee` without pipefail).
require_env() {
  python -c 'import stable_worldmodel, h5py, hdf5plugin' 2>/dev/null || {
    echo "[FATAL] python deps missing (stable_worldmodel/h5py); setup_local_env.sh did not finish."
    echo "        Rerun: bash batch/setup_local_env.sh   and read its output for the first error."
    exit 1
  }
}
require_h5() {
  [ -n "$(H5)" ] || {
    echo "[FATAL] no .h5 dataset found under $DATA_DIR; run the download stage first (and check it COMPLETED)."
    exit 1
  }
}

# ${REACHER_GPU_DEVICES:-} left unset/empty is intentional (see fan_out below).
DEVICES=(${REACHER_GPU_DEVICES:-})

# Runs a list of commands (one per line on stdin) either through the MIG
# worker pool (if DEVICES is non-empty) or plain sequential bash -c.
# REACHER_SHARD="k/N" (e.g. 0/2, 1/2) keeps only every N-th job starting at k
# which lets two/three boxes with a SHARED home split one stage's job list
# without any coordination beyond agreeing on k. Job list order is
# deterministic, so shards are disjoint and complete by construction.
fan_out_pool() {
  if [ "${#DEVICES[@]}" -gt 0 ]; then
    bash batch/dispatch_pool.sh "${DEVICES[@]}"
  else
    while IFS= read -r cmd; do
      [ -z "$cmd" ] && continue
      bash -c "$cmd"
    done
  fi
}

fan_out() {
  if [ -n "${REACHER_SHARD:-}" ]; then
    local k=${REACHER_SHARD%%/*} n=${REACHER_SHARD##*/}
    echo "[shard] running shard $k of $n (every ${n}th job starting at index $k)"
    awk -v k="$k" -v n="$n" '(NR-1)%n==k' | fan_out_pool
  else
    fan_out_pool
  fi
}

# fetch <url> <outfile>: resumable download; curl > wget > pure python
# (some of these boxes ship with NEITHER curl nor wget; python always exists).
fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl -L -C - --retry 10 --retry-delay 15 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$2" "$1"
  else
    local i
    for i in $(seq 1 10); do
      python - "$1" "$2" <<'PY' && return 0
import os, sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
pos = os.path.getsize(out) if os.path.exists(out) else 0
req = urllib.request.Request(url, headers={"User-Agent": "fetch/1.0"})
if pos:
    req.add_header("Range", f"bytes={pos}-")
r = urllib.request.urlopen(req, timeout=60)
mode = "ab" if pos and getattr(r, "status", 200) == 206 else "wb"
if mode == "wb":
    pos = 0
done = 0
with open(out, mode) as f:
    while True:
        chunk = r.read(1 << 22)
        if not chunk:
            break
        f.write(chunk)
        done += len(chunk)
        print(f"\r  {(pos + done) / 1e9:.2f} GB", end="", flush=True)
print()
PY
      echo "[fetch] attempt $i failed; retrying in 15s (resumes from partial file)"; sleep 15
    done
    return 1
  fi
}

run_download() {
  require_env
  if [ -n "$(H5)" ] && [ -f encoder_reacher/weights.pt ]; then
    echo "[download] dataset + encoder already present, skipping"
    return
  fi
  echo "=== downloading reacher.tar.zst (23.8GB, resumable; reruns continue where they left off) ==="
  ( cd "$DATA_DIR" && \
    fetch https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst reacher.tar.zst && \
    echo "=== download complete, extracting ===" && \
    tar --zstd -xf reacher.tar.zst && \
    rm reacher.tar.zst && ls -la )
  mkdir -p encoder_reacher
  ( cd encoder_reacher && \
    fetch https://huggingface.co/quentinll/lewm-reacher/resolve/main/config.json config.json && \
    fetch https://huggingface.co/quentinll/lewm-reacher/resolve/main/weights.pt weights.pt )
  echo "=== inspect ==="
  python batch/inspect_reacher.py 2>&1 | tee logs/inspect_reacher.log
  echo "[download] check logs/inspect_reacher.log BEFORE trusting anything downstream"
  echo "           (schema: qpos/qvel/pixels/action + ep_offset/ep_len/episode_idx/step_idx)"
}

run_smoke() {
  require_env; require_h5
  local h5; h5=$(H5)
  echo "=== [1/2] ORACLE n=32 (harness + encoder + injection sanity) ==="
  python -m specaccept.envs.reacher.eval --subgoal oracle \
    --source local --local-dir encoder_reacher --h5 "$h5" --device cuda \
    --num-eval 32 --seed 42 --episode-min 8000 2>&1 | tee logs/reacher_smoke_oracle.log
  echo "=== [2/2] BASELINE n=32 (flat LeWM, goal image == LeWM protocol) ==="
  python -m specaccept.envs.reacher.eval --subgoal baseline \
    --source local --local-dir encoder_reacher --h5 "$h5" --device cuda \
    --num-eval 32 --seed 42 --episode-min 8000 2>&1 | tee logs/reacher_smoke_baseline.log
  echo "[smoke] check both logs: baseline SR well above 0, oracle >= baseline, before spending training compute"
}

run_dense() {
  require_env; require_h5
  local h5; h5=$(H5)
  echo "=== dense build from $h5 ==="
  python -m specaccept.envs.reacher.build_subgoals --source local --local-dir encoder_reacher \
    --h5 "$h5" --out subgoals_reacher_dense.pt --stride 1 --device cuda --batch-size 256
}

run_train() {
  require_env
  [ -f subgoals_reacher_dense.pt ] || { echo "[FATAL] subgoals_reacher_dense.pt missing; run the dense stage first."; exit 1; }
  python - <<'PY'
import torch, numpy as np
blob = torch.load('subgoals_reacher_dense.pt', map_location='cpu', weights_only=False)
ep = blob['episode_idx'].numpy()
mask = ep < 8000
np.save('reacher_train_mask.npy', mask)
print(f'[mask] train episodes: {mask.sum()}/{len(mask)} (holdout = {int((~mask).sum())})')
PY
  local cmds=()
  for S in 5 10 15 25; do
    cmds+=("echo '=== train gdm_reacher_s${S}.pt ==='; python -m specaccept.train_drafter --subgoals subgoals_reacher_dense.pt --out gdm_reacher_s${S}.pt --device cuda --mask custom --custom-mask-file reacher_train_mask.npy --subgoal-step ${S} --goal-cond --goal-rule window --goal-gap 1 --goal-gap-max 100 --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp 2>&1 | tee logs/reacher_train_s${S}.log")
  done
  printf '%s\n' "${cmds[@]}" | fan_out
}

run_arms() {
  require_env; require_h5
  for c in gdm_reacher_s5.pt gdm_reacher_s10.pt gdm_reacher_s15.pt gdm_reacher_s25.pt; do
    [ -f "$c" ] || { echo "[FATAL] $c missing; run the train stage first (or scp checkpoints from the training box)."; exit 1; }
  done
  local h5; h5=$(H5)
  local NAMES=(baseline s25gdm s15gdm s10gdm s5gdm s25spec s15spec s10spec s5spec)
  local SUBGOALS=(baseline gdm gdm gdm gdm specaccept specaccept specaccept specaccept)
  local CKPTS=(none gdm_reacher_s25.pt gdm_reacher_s15.pt gdm_reacher_s10.pt gdm_reacher_s5.pt gdm_reacher_s25.pt gdm_reacher_s15.pt gdm_reacher_s10.pt gdm_reacher_s5.pt)
  local RHS=(5 5 3 2 1 5 3 2 1)
  local KS=(50 50 50 50 50 8 8 8 8)
  local SEEDS=(42 43 44 45)
  local cmds=()
  for c in "${!NAMES[@]}"; do
    local NAME=${NAMES[$c]} SG=${SUBGOALS[$c]} CKPT=${CKPTS[$c]} RH=${RHS[$c]} K=${KS[$c]}
    local CKPT_ARG=""
    [ "$CKPT" != "none" ] && CKPT_ARG="--gdm-ckpt $CKPT"
    for SEED in "${SEEDS[@]}"; do
      cmds+=("echo '=== arm=$NAME seed=$SEED ==='; python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} --accept-tau 0.20 --gdm-steps ${K} --source local --local-dir encoder_reacher --h5 $h5 --device cuda --num-eval 128 --seed ${SEED} --episode-min 8000 --goal-offset 25 --eval-budget 50 --horizon ${RH} --receding-horizon ${RH} --dump-traces runs/reacher/traces_${NAME}_seed${SEED}.pt 2>&1 | tee logs/reacher_arms_${NAME}_seed${SEED}.log")
    done
  done
  printf '%s\n' "${cmds[@]}" | fan_out
}

run_horizon() {
  require_env; require_h5
  for c in gdm_reacher_s10.pt gdm_reacher_s25.pt; do
    [ -f "$c" ] || { echo "[FATAL] $c missing; run the train stage first (or scp checkpoints from the training box)."; exit 1; }
  done
  local h5; h5=$(H5)
  local ARMS=(s25gdm s10gdm s10spec)
  local SUBGOALS=(gdm gdm specaccept)
  local CKPTS=(gdm_reacher_s25.pt gdm_reacher_s10.pt gdm_reacher_s10.pt)
  local RHS=(5 2 2)
  local KS=(50 50 8)
  local OFFSETS=(25 50 75 100)
  local SEEDS=(42 43)
  # Pre-generate all (arm,offset) episode files sequentially FIRST; avoids a
  # write race if two parallel seed-jobs for the same (arm,offset) both try to
  # (re)generate the same EPFILE at once. Content is deterministic (seed 42)
  # so this is a pure dedup, not a correctness fix beyond race-avoidance.
  for a in "${!ARMS[@]}"; do
    for OFF in "${OFFSETS[@]}"; do
      local EPFILE="reacher_horizon.ep.${ARMS[$a]}_t${OFF}.json"
      python batch/build_reacher_horizon_episodes.py --h5 "$h5" --out "$EPFILE" \
        --n 128 --max-offset 100 --episode-min 8000 --seed 42
    done
  done
  local cmds=()
  for a in "${!ARMS[@]}"; do
    local NAME=${ARMS[$a]} SG=${SUBGOALS[$a]} CKPT=${CKPTS[$a]} RH=${RHS[$a]} K=${KS[$a]}
    for OFF in "${OFFSETS[@]}"; do
      local EPFILE="reacher_horizon.ep.${NAME}_t${OFF}.json"
      for SEED in "${SEEDS[@]}"; do
        local cmd="echo '=== arm=$NAME offset=$OFF seed=$SEED (budgets: 2t=$((2*OFF)), vlwm=50) ==='"
        cmd+="; echo '--- FF-JEPA regime: budget 2t ---'"
        cmd+="; python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} --accept-tau 0.20 --gdm-steps ${K} --source local --local-dir encoder_reacher --h5 $h5 --device cuda --num-eval 128 --seed ${SEED} --episodes-file ${EPFILE} --goal-offset ${OFF} --eval-budget $((2*OFF)) --horizon ${RH} --receding-horizon ${RH} --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_2t_seed${SEED}.pt 2>&1 | tee logs/reacher_horizon_${NAME}_t${OFF}_seed${SEED}.log"
        if [ "${OFF}" -ne 25 ]; then
          cmd+="; echo '--- VLWM regime: budget 50 fixed ---'"
          cmd+="; python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} --accept-tau 0.20 --gdm-steps ${K} --source local --local-dir encoder_reacher --h5 $h5 --device cuda --num-eval 128 --seed ${SEED} --episodes-file ${EPFILE} --goal-offset ${OFF} --eval-budget 50 --horizon ${RH} --receding-horizon ${RH} --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_b50_seed${SEED}.pt 2>&1 | tee -a logs/reacher_horizon_${NAME}_t${OFF}_seed${SEED}.log"
        fi
        cmds+=("$cmd")
      done
    done
  done
  printf '%s\n' "${cmds[@]}" | fan_out
}

case "$STAGE" in
  download) run_download ;;
  smoke)    run_smoke ;;
  dense)    run_dense ;;
  train)    run_train ;;
  arms)     run_arms ;;
  horizon)  run_horizon ;;
  all)
    run_download
    run_smoke
    run_dense
    run_train
    run_arms
    run_horizon
    ;;
  *) echo "unknown stage: $STAGE (want: download|smoke|dense|train|arms|horizon|all)"; exit 1 ;;
esac
echo "=== stage '$STAGE' done ==="
