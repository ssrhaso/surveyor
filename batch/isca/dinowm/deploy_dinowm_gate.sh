#!/usr/bin/env bash
# Installs the arrival-gate serving stack into dino_wm and teaches the runner
# the GOAL_GATE / ARRIVE_TAU environment variables. Idempotent.
set -euo pipefail
DWM=/lustre/home/ha676/dino_wm
SRC=/lustre/home/ha676/le-wm/batch/isca/dinowm

cp "$SRC/specaccept_dinowm.py" "$DWM/specaccept_dinowm.py"
echo "[deploy] specaccept_dinowm.py (arrival gate) installed"

python3 - "$DWM/run_plan_pusht_spec.py" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
if "GOAL_GATE" in src:
    print("[deploy] runner already knows GOAL_GATE")
    raise SystemExit
anchor = '    cfg["planner"]["spec_seed"] = cfg["seed"]\n'
if anchor not in src:
    raise SystemExit("[deploy] FAILED: anchor line not found in runner")
add = anchor + (
    '    if os.environ.get("GOAL_GATE"):\n'
    '        cfg["planner"]["goal_gate"] = True\n'
    '    if os.environ.get("ARRIVE_TAU"):\n'
    '        cfg["planner"]["arrive_tau"] = float(os.environ["ARRIVE_TAU"])\n'
)
open(p, "w").write(src.replace(anchor, add))
print("[deploy] runner patched for GOAL_GATE / ARRIVE_TAU")
PY
