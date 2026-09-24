#!/usr/bin/env bash
# Run one row of a manifest, from the repository root:
#   bash reproduce/run_manifest.sh reproduce/battery_A.tsv 17
# Rows are tab-separated: idx env arm t seed log cmd. The command's output goes to the row's log; a row whose log
# already carries a result is skipped, so a manifest can be re-run and only the missing rows run.
# Environment: DATA, the directory holding the datasets (pusht_expert_train.h5, reacher.h5, cube_single_expert.h5,
# tworoom.h5); REACHER_H5 and CUBE_H5 default to the files in DATA.
set -uo pipefail
MAN=${1:?usage: run_manifest.sh <manifest.tsv> <row idx>}
ID=${2:?usage: run_manifest.sh <manifest.tsv> <row idx>}
: "${DATA:?set DATA to the directory that holds the datasets}"
export DATA
export REACHER_H5=${REACHER_H5:-$DATA/reacher.h5}
export CUBE_H5=${CUBE_H5:-$DATA/cube_single_expert.h5}
export MUJOCO_GL=${MUJOCO_GL:-egl} SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} PYTHONUNBUFFERED=1
LINE=$(awk -F'\t' -v id="$ID" 'NR > 1 && $1 == id' "$MAN")
[ -n "$LINE" ] || { echo "no row $ID in $MAN"; exit 1; }
IFS=$'\t' read -r IDX ENV ARM T SEED LOG CMD <<< "$LINE"
mkdir -p "$(dirname "$LOG")"
if [ -f "$LOG" ] && grep -q -E "\[RESULT\]|SR ?= ?[0-9]" "$LOG"; then
  echo "already done: $LOG"; exit 0
fi
echo "ROW idx=$IDX env=$ENV arm=$ARM t=$T seed=$SEED"
echo "CMD: $CMD"
eval "$CMD" 2>&1 | tee "$LOG.part"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ] && grep -q -E "\[RESULT\]|SR ?= ?[0-9]" "$LOG.part"; then
  mv "$LOG.part" "$LOG"; echo "done: $LOG"
else
  echo "FAILED idx=$IDX rc=$rc (partial log kept at $LOG.part)"; exit 1
fi
