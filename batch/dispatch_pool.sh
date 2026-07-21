#!/usr/bin/env bash
# Dynamic worker pool across MIG slices (or plain GPU indices). Reads shell
# commands from stdin, one per line; each is dispatched to a free device via
# CUDA_VISIBLE_DEVICES the moment one becomes free (not static round-robin --
# a fast job's slice is reused immediately, not left idle until a batch ends).
#
# Usage:
#   printf '%s\n' "${cmds[@]}" | bash batch/dispatch_pool.sh MIG-xxx MIG-yyy ...
set -eu
DEVICES=("$@")
[ "${#DEVICES[@]}" -gt 0 ] || { echo "usage: dispatch_pool.sh <device> [<device> ...] < commands.txt" >&2; exit 1; }

FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm -f "$FIFO"
for d in "${DEVICES[@]}"; do echo "$d" >&3; done

PIDS=()
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  read -u 3 dev
  (
    echo "[dispatch] device=$dev :: $cmd"
    CUDA_VISIBLE_DEVICES="$dev" bash -c "$cmd"
    status=$?
    echo "$dev" >&3
    exit $status
  ) &
  PIDS+=("$!")
done

fail=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || fail=1
done
exec 3>&-
[ "$fail" -eq 0 ] || { echo "[dispatch_pool] one or more jobs failed; check logs" >&2; exit 1; }
