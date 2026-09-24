"""Harvest one battery from its logs only.

Reads logs/<prefix>_{env}_{arm}_t{t}_seed{seed}.log under the current directory, takes the operative success rate
(20-degree block criterion on PushT/Reacher, the plain SR elsewhere) and the last call_ratio, pools seeds without
selection, and writes <prefix>_cells.csv plus a per-environment grid <prefix>_grid.md next to this script (or into
$HARVEST_OUT when it is set).

    python reproduce/harvest_rbA.py rbA          # from the directory that holds logs/
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np

SR20 = re.compile(r"\[RESULT\][^\n]*angle=20deg\s+SR=([0-9.]+)%")
SRC = re.compile(r"SR\s*=\s*([0-9.]+)%")
CR = re.compile(r"call_ratio=([0-9.]+)")
# the fixed-depth blind rows (dspark commit-k) print drafts per replanning decision as
# 'redraft/advance=' (re-drafts / all decisions), the same ratio the other rows print as call_ratio
CR2 = re.compile(r"redraft/advance=([0-9.]+)")
import sys
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "rbA"
OUT = os.environ.get("HARVEST_OUT", os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(PREFIX + r"_(pusht|reacher|cube|tworoom)_([A-Za-z0-9_.]+)_t(\d+)_seed(\d+)\.log$")
ARM_ORDER = ["flat_rh5", "flat_rh2", "ffjepa_s25", "ffjepa_gf", "surveyor", "router",
             "gcidm", "gcidm_surv", "gcidm_every", "gcidm_blind"]


def parse(path):
    with open(path, errors="ignore") as fh:
        t = fh.read()
    m = SR20.search(t)
    sr = float(m.group(1)) if m else None
    if sr is None:
        ms = SRC.findall(t)
        sr = float(ms[-1]) if ms else None
    cr = CR.findall(t) or CR2.findall(t)
    return sr, (float(cr[-1]) if cr else None)


def main():
    cells = defaultdict(dict)
    ratios = defaultdict(dict)
    for path in sorted(glob.glob(f"logs/{PREFIX}_*_t*_seed*.log")):
        m = NAME.search(path)
        if not m:
            continue
        env, arm, t, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        sr, cr = parse(path)
        if sr is None:
            continue
        cells[(env, arm, t)][seed] = sr
        if cr is not None:
            ratios[(env, arm, t)][seed] = cr

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for (env, arm, t), d in sorted(cells.items(), key=lambda kv: (kv[0][0], ARM_ORDER.index(kv[0][1]) if kv[0][1] in ARM_ORDER else 99, kv[0][2])):
        seeds = sorted(d)
        v = np.array([d[s] for s in seeds])
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        r = ratios.get((env, arm, t), {})
        cr = float(np.mean([r[s] for s in r])) if r else None
        rows.append((env, arm, t, len(seeds), "|".join(map(str, seeds)), float(v.mean()), se,
                     "|".join(f"{x:.2f}" for x in v), "" if cr is None else f"{cr:.3f}"))
    with open(f"{OUT}/{PREFIX}_cells.csv", "w") as f:
        f.write("env,arm,t,n_seeds,seeds,sr_mean_pct,sr_se_pct,sr_per_seed,call_ratio\n")
        for row in rows:
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.2f}" for x in row) + "\n")

    lines = [f"# {PREFIX}, pooled over seeds 42-49, no selection", ""]
    DEFAULT_TS = {"pusht": [25, 50, 75, 100, 150], "reacher": [25, 50, 100, 150], "cube": [150], "tworoom": [25, 50, 75, 100]}
    for env in ("pusht", "reacher", "cube", "tworoom"):
        ts = sorted(set(DEFAULT_TS[env]) | {k[2] for k in cells if k[0] == env})  # columns follow the data
        present = {k[1] for k in cells if k[0] == env}
        arms = [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))
        if not arms:
            continue
        lines.append(f"## {env}  (SR %, n seeds in brackets; call ratio for executor rows)")
        lines.append("| arm | " + " | ".join(f"t={t}" for t in ts) + " |")
        lines.append("|---|" + "---|" * len(ts))
        for a in arms:
            out = []
            for t in ts:
                d = cells.get((env, a, t))
                if not d:
                    out.append("-")
                    continue
                v = np.array(list(d.values()))
                r = ratios.get((env, a, t), {})
                tag = f" cr {np.mean(list(r.values())):.2f}" if r else ""
                out.append(f"{v.mean():.2f} [{len(v)}]{tag}")
            lines.append(f"| {a} | " + " | ".join(out) + " |")
        lines.append("")
    with open(f"{OUT}/{PREFIX}_grid.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT}/{PREFIX}_cells.csv ({len(rows)} cells) and {OUT}/{PREFIX}_grid.md")


if __name__ == "__main__":
    main()
