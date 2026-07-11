#!/usr/bin/env bash
# One-shot Reacher-battery deployment to ISCA (Exeter) from the Windows laptop
# (Git Bash):   bash batch/isca/deploy_reacher_isca.sh
#
# ISCA sibling of batch/deploy_reacher_isambard.sh, but simpler: plain ed25519
# key auth (Host "isca" in ~/.ssh/config, no clifton cert to expire) and no
# apptainer. Prerequisites on ISCA (one-time, already scripted):
#   ~/bootstrap_venv.sh   -> ~/le-wm/.venv          (BOOTSTRAP-DONE in ~/bootstrap.log)
#   ~/download_reacher.sh -> ~/data/reacher + encoder (DOWNLOAD-DONE in ~/download.log)
# TwoRoom is NOT deployable here: its prerequisites (crossroom dense latents,
# encoder_tworoom) exist only on the Isambard box from the parked phase.
set -eu
SSH='/c/Windows/System32/OpenSSH/ssh.exe'
HOST=isca
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== [1/3] sync specaccept package + batch scripts ==="
# ship the whole package tree; drop any stale copy so old module paths cannot resolve
tar -C "$LOCAL" --exclude='__pycache__' -cf - specaccept \
  | "$SSH" -o BatchMode=yes "$HOST" 'mkdir -p le-wm && rm -rf le-wm/specaccept && tar -C le-wm -xf -'
tar -C "$LOCAL" -cf - \
  batch/isca \
  batch/inspect_reacher.py \
  batch/build_reacher_horizon_episodes.py \
  batch/aggregate_reacher_results.py \
  | "$SSH" -o BatchMode=yes "$HOST" 'tar -C le-wm -xf -'

echo "=== [2/3] submit chain ==="
"$SSH" -o BatchMode=yes "$HOST" 'cd ~/le-wm && bash batch/isca/submit_reacher_chain.sh'

echo "=== [3/3] current queue ==="
"$SSH" -o BatchMode=yes "$HOST" 'squeue --me'
