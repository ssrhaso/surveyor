#!/usr/bin/env bash
#SBATCH --job-name=stride-gate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_gate_%j.log
# Stage 2 mechanical gate (stride-spacing round). Runs after the train array
# (submit with: sbatch --dependency=afterany:<ARRAYID> batch/run_stride_gate.sbatch <ARRAYID>)
# and parses each arm's in-job diag_gdm output. Gates OUT (never greenlights
# quality -- rel_err does not decide SR, validated inversion):
#   * degeneracy:  B rel_err < 0.5 x noop_err for every stride
#   * collapse:    B collapse >= 0.6
#   * S=25 sanity: B-success rel_err in [0.10, 0.25] (must reproduce
#     gdm_faithful.pt's ~0.16, proving the rebuilt-dense + retrain pipeline)
# All pass -> auto-submits the Stage 3 closed-loop A/B (run_stride_ab.sbatch).
# Any fail -> writes logs/STRIDE_GATE_FAILED.txt and submits nothing.
set -eu
cd /home/u6ko/hasoshu.u6ko/le-wm
ARRAYID=$1
python3 - "$ARRAYID" <<'EOF'
import glob, re, subprocess, sys

arrayid = sys.argv[1]
logs = sorted(glob.glob(f"logs/stride_train_{arrayid}_*.log"))
print(f"[gate] array {arrayid}: found {len(logs)} arm logs: {logs}")

def parse(path):
    txt = open(path).read()
    m_s = re.search(r"subgoal_step[= ](\d+)", txt)
    blocks = {}
    # every show() block: [TAG] then rel_err/collapse lines
    for m in re.finditer(r"\[([^\]\n]+)\]\n  rel_err  = ([\d.]+)\s+\(vs no-op baseline ([\d.]+).*?collapse = ([\d.]+)", txt, re.S):
        blocks[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return (int(m_s.group(1)) if m_s else None), blocks

ok, reasons = True, []
seen = {}
for p in logs:
    S, blocks = parse(p)
    bsucc = next((v for k, v in blocks.items() if k.startswith("B-success")), None)
    b = next((v for k, v in blocks.items() if k.startswith("B:")), None)
    pick = bsucc or b
    if S is None or pick is None:
        ok = False; reasons.append(f"{p}: no parsable diag output (train crashed?)"); continue
    rel, noop, col = pick
    seen[S] = pick
    print(f"[gate] S={S}: rel_err={rel:.4f} noop={noop:.4f} collapse={col:.3f} ({p})")
    if rel >= 0.5 * noop:
        ok = False; reasons.append(f"S={S}: DEGENERATE rel_err {rel:.3f} >= 0.5*noop {noop:.3f}")
    if col < 0.6:
        ok = False; reasons.append(f"S={S}: collapse {col:.3f} < 0.6")
missing = {5, 10, 15, 25} - set(seen)
if missing:
    ok = False; reasons.append(f"missing strides {sorted(missing)}")
if 25 in seen and not (0.10 <= seen[25][0] <= 0.25):
    ok = False; reasons.append(f"S=25 sanity FAIL: rel_err {seen[25][0]:.4f} outside [0.10,0.25] "
                               f"(gdm_faithful ~0.16) -- rebuild/retrain pipeline not faithful")

if ok:
    print("[gate] ALL PASS -> submitting Stage 3 closed-loop A/B")
    out = subprocess.run(["sbatch", "batch/run_stride_ab.sbatch"], capture_output=True, text=True)
    print(out.stdout, out.stderr)
else:
    with open("logs/STRIDE_GATE_FAILED.txt", "w") as f:
        f.write("\n".join(reasons) + "\n")
    print("[gate] FAILED -- Stage 3 NOT submitted:\n  " + "\n  ".join(reasons))
EOF
