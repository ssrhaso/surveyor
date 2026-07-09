#!/usr/bin/env bash
# One-shot generalization-phase deployment (Reacher + TwoRoom) from the
# Windows laptop (Git Bash), for when Isambard comes back up:
#   bash batch/deploy_reacher_isambard.sh            # both chains
#   bash batch/deploy_reacher_isambard.sh reacher    # reacher only
#   bash batch/deploy_reacher_isambard.sh tworoom    # tworoom only
# Uses the WINDOWS OpenSSH binaries (the Windows ssh-agent holds the unlocked
# key; Git Bash's own ssh cannot see it). If auth fails with the agent loaded,
# the ~12h clifton cert has expired - re-run clifton login first.
set -eu
WHAT=${1:-all}
SSH='/c/Windows/System32/OpenSSH/ssh.exe'
SCP='/c/Windows/System32/OpenSSH/scp.exe'
CFG='C:\Users\hasaa\.ssh\config_clifton'
HOST=u6ko.aip2.isambard
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== [1/3] sync ffjepa modules + batch scripts ==="
"$SCP" -F "$CFG" -o BatchMode=yes \
  "$LOCAL/ffjepa/build_subgoals_reacher.py" \
  "$LOCAL/ffjepa/eval_reacher.py" \
  "$LOCAL/ffjepa/build_subgoals_tworoom.py" \
  "$LOCAL/ffjepa/eval_tworoom.py" \
  "$LOCAL/ffjepa/train_gdm.py" \
  "$LOCAL/ffjepa/subgoal_planner.py" \
  "$LOCAL/ffjepa/eval_ffjepa.py" \
  "$HOST":le-wm/ffjepa/
"$SCP" -F "$CFG" -o BatchMode=yes \
  "$LOCAL/batch/run_reacher_download_inspect.sbatch" \
  "$LOCAL/batch/download_inspect_reacher.sh" \
  "$LOCAL/batch/inspect_reacher.py" \
  "$LOCAL/batch/run_reacher_dense_build.sbatch" \
  "$LOCAL/batch/run_reacher_smoke.sbatch" \
  "$LOCAL/batch/run_reacher_train_array.sbatch" \
  "$LOCAL/batch/run_reacher_eval_arms.sbatch" \
  "$LOCAL/batch/run_reacher_horizon_sweep.sbatch" \
  "$LOCAL/batch/build_reacher_horizon_episodes.py" \
  "$LOCAL/batch/submit_reacher_chain.sh" \
  "$LOCAL/batch/run_tworoom_train_array.sbatch" \
  "$LOCAL/batch/run_tworoom_eval_arms.sbatch" \
  "$LOCAL/batch/submit_tworoom_chain.sh" \
  "$HOST":le-wm/batch/

echo "=== [2/3] submit chains ($WHAT) ==="
if [ "$WHAT" = "all" ] || [ "$WHAT" = "reacher" ]; then
  "$SSH" -F "$CFG" -o BatchMode=yes "$HOST" 'cd ~/le-wm && bash batch/submit_reacher_chain.sh'
fi
if [ "$WHAT" = "all" ] || [ "$WHAT" = "tworoom" ]; then
  "$SSH" -F "$CFG" -o BatchMode=yes "$HOST" 'cd ~/le-wm && bash batch/submit_tworoom_chain.sh'
fi

echo "=== [3/3] current queue ==="
"$SSH" -F "$CFG" -o BatchMode=yes "$HOST" 'squeue --me'
