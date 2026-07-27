#!/usr/bin/env bash
# Installs the TwoRoom eval wrapper into dino_wm and validates it against the
# recorded data. Idempotent: safe to re-run after editing the patches.
# Usage (from le-wm on ISCA):  bash batch/isca/dinowm/deploy_tworoom_eval.sh
set -euo pipefail
DWM=/lustre/home/ha676/dino_wm
# patches are staged next to this script (ISCA keeps the dinowm runners in
# $DWM; the committed copies live in le-wm/batch/isca/dinowm/)
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DWM/env/tworoom"
cp "$SRC/tworoom_env_wrapper.py" "$DWM/env/tworoom/tworoom_env_wrapper.py"
: > "$DWM/env/tworoom/__init__.py"
[ "$SRC/validate_tworoom_env.py" -ef "$DWM/validate_tworoom_env.py" ] || \
  cp "$SRC/validate_tworoom_env.py" "$DWM/validate_tworoom_env.py"

# --- register the gym id (idempotent append) --------------------------------
python3 - "$DWM/env/__init__.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
if "tworoom_dino" in src:
    print("[deploy] gym id already registered")
else:
    src += '''
register(
    id="tworoom_dino",
    entry_point="env.tworoom.tworoom_env_wrapper:TwoRoomEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
'''
    open(p, "w").write(src)
    print("[deploy] registered tworoom_dino")
PY

# --- plan.py must use the serial vector env (torch renderer, no subprocs) ---
python3 - "$DWM/plan.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = 'if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":'
new = ('if model_cfg.env.name in ("wall", "deformable_env", "tworoom_dino"):')
if new in src:
    print("[deploy] plan.py already patched")
elif old in src:
    open(p, "w").write(src.replace(old, new))
    print("[deploy] plan.py -> SerialVectorEnv for tworoom_dino")
else:
    raise SystemExit("[deploy] FAILED: plan.py vector-env branch not found")
PY

echo "=== validating vendored env against tworoom.h5 ==="
cd "$DWM"
.venv/bin/python validate_tworoom_env.py
