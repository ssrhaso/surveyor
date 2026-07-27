#!/usr/bin/env bash
# Stages the TwoRoom world model into the layout plan.py expects:
#   $CKPT/outputs/tworoom_dino/{hydra.yaml, checkpoints/model_latest.pth}
#
# Usage:
#   bash stage_tworoom_ckpt.sh live     # symlink the live training ckpt (smoke)
#   bash stage_tworoom_ckpt.sh freeze   # hard copy (every battery arm shares
#                                       # one identical WM -- required before
#                                       # any pre-registered cell)
set -euo pipefail
MODE=${1:-live}
DWM=/lustre/home/ha676/dino_wm
CKPT=/lustre/home/ha676/data/dinowm/checkpoints/outputs/tworoom_dino

# newest training run that actually produced a checkpoint
RUN=$(ls -d $DWM/outputs/*/*/ 2>/dev/null | while read -r d; do
        [ -f "$d/checkpoints/model_latest.pth" ] && \
        [ -f "$d/.hydra/config.yaml" ] && \
        grep -q "tworoom" "$d/.hydra/config.yaml" && echo "$d"
      done | tail -1)
[ -n "$RUN" ] || { echo "FAILED: no tworoom training run with a checkpoint"; exit 1; }
echo "[stage] run   = $RUN"

mkdir -p "$CKPT/checkpoints"
cp "$RUN/.hydra/config.yaml" "$CKPT/hydra.yaml"
rm -f "$CKPT/checkpoints/model_latest.pth"
if [ "$MODE" = "freeze" ]; then
  cp "$RUN/checkpoints/model_latest.pth" "$CKPT/checkpoints/model_latest.pth"
  echo "[stage] FROZEN copy"
else
  ln -s "$RUN/checkpoints/model_latest.pth" "$CKPT/checkpoints/model_latest.pth"
  echo "[stage] live symlink"
fi

# the training epoch actually being served -- goes in every log for the record
$DWM/.venv/bin/python - "$CKPT/checkpoints/model_latest.pth" <<'PY'
import sys
import torch
p = torch.load(sys.argv[1], map_location="cpu")
print("[stage] serving epoch:", p.get("epoch"))
PY
ls -laL "$CKPT/checkpoints/"
