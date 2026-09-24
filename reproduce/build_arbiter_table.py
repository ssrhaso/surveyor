"""What the arbiter does: per full-layer cell, the fraction of episodes routed to the executor at
the first replanning boundary and the fraction retired later.

Sources (the arbiter banner line of each router log, as extract_banners.py writes them):
  arbiter_banners_A.txt      '[router+surveyor] tau=.. retired=A/N (fired-at-first-replan=B, ...' lines of the
                             rbA/rbH/rbK router logs of cluster A (PushT logs print the banner twice; the last line per
                             log is the final one)
  arbiter_banners_B.txt      the same lines for rbL/rbM/rbN, cluster B (gd_router and the arbiter variants)
  gcrouter_banners_A.txt     '[gcidm+surveyor cost] ... routed=R arrived=V ...' of the official GC-IDM router logs
  arbiter_banners_pulled.txt, gcrouter_banners_pulled.txt   the same lines of the later batteries
  arbiter_banners_TR*.txt    Two-Room batteries with the arbiter read in LeWM space
Definitions: routed = B/N (search arms) or R/N (GC-IDM): the episode never drafted.
             retired later = (A-B)/N or V/N: drafting started and was switched off at a later boundary.
Writes arbiter_activity.csv (every distance) next to this script and, into the directory given as the first argument
(default out/tables), table1_arbiter.tex (the shortest and the longest distance of each environment).

    python reproduce/build_arbiter_table.py [out_dir]
"""
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_table1 as b  # noqa: E402

HERE = b.HERE
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "tables")
NAME = re.compile(r"^(rb[A-Za-z0-9]+)_(pusht|reacher|cube|tworoom)_(.+)_t(\d+)_seed(\d+)\.log$")
SEARCH = re.compile(r"retired=(\d+)/(\d+) \(fired-at-first-replan=(\d+)")
GCIDM = re.compile(r"routed=(\d+) arrived=(\d+)")
N_EP = {"pusht": 256, "reacher": 128, "cube": 128, "tworoom": 64}
FULL_ARMS = [("CEM (LeWM)", "router"), ("GC-IDM (released code)", "gcrouter_off_s42"), ("iCEM", "icem_router"),
             ("MPPI", "mppi_router"), ("Gradient (AdamW)", "gd_router")]
ENV_NAME = {"pusht": "PushT", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room"}


def load():
    last = {}   # filename -> (routed, later, n)
    for fn in ("arbiter_banners_A.txt", "arbiter_banners_B.txt", "gcrouter_banners_A.txt",
               "arbiter_banners_pulled.txt", "gcrouter_banners_pulled.txt",   # extract_banners.py, later batteries
               "arbiter_banners_TR.txt"):   # battery TR: Two-Room, arbiter read in LeWM space
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "\t" not in line:
                    continue
                name, text = line.rstrip("\n").split("\t", 1)
                m = NAME.match(name)
                if not m:
                    continue
                env = m.group(2)
                s = SEARCH.search(text)
                g = GCIDM.search(text)
                if s:
                    a, n, fired = int(s.group(1)), int(s.group(2)), int(s.group(3))
                    last[name] = (fired / n, (a - fired) / n, n)
                elif g:
                    n = N_EP[env]
                    last[name] = (int(g.group(1)) / n, int(g.group(2)) / n, n)
    cells = defaultdict(dict)   # (pref, env, arm, t) -> {seed: (routed, later)}
    for name, (r, l, n) in last.items():
        pref, env, arm, t, seed = NAME.match(name).groups()
        cells[(pref, env, arm, int(t))][int(seed)] = (r, l)
    return cells


def get(cells, env, arm, t):
    # battery TR replaces the Two-Room full rows, as in build_table1.get
    if env == "tworoom" and b.TR_ON:
        d = cells.get((b.TR_PREF, env, arm + b.TR_SUFFIX, t))
        if d and len(d) >= 8:
            return d
    # as in build_table1.get: the full rows the paper prints are the runs at the environment's own tolerance (arm
    # suffix _tauown, batteries T / U), so their banners describe the arbiter, not those of the 0.20 runs
    for p in b.OWN_PREFS:
        d = cells.get((p, env, arm + b.OWN_SUFFIX, t))
        if d and len(d) >= 8:
            return d
    best = None
    for p in b.ENV_PREFS.get(env, b.PREFS):
        d = cells.get((p, env, arm, t))
        if not d:
            continue
        if len(d) >= 8:
            return d
        if best is None or len(d) > len(best):
            best = d
    return best


def main():
    cells = load()
    rows = []
    for exe, arm in FULL_ARMS:
        for env in b.ENVS:
            for t in b.T[env]:
                d = get(cells, env, arm, t)
                if not d:
                    continue
                r = 100 * sum(v[0] for v in d.values()) / len(d)
                l = 100 * sum(v[1] for v in d.values()) / len(d)
                rows.append((exe, arm, env, t, len(d), r, l))
    with open(os.path.join(HERE, "arbiter_activity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["executor", "arm", "env", "t", "n_seeds", "routed_pct", "retired_later_pct"])
        for row in rows:
            w.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.1f}", f"{row[6]:.1f}"])
    look = {(r[0], r[2], r[3]): (r[5], r[6], r[4]) for r in rows}
    cols = [(e, t) for e in b.ENVS for t in (min(b.T[e]), max(b.T[e]))]
    lines = ["% generated by reproduce/build_arbiter_table.py; do not edit by hand",
             "\\begin{tabular}{@{}ll" + "rr" * len(b.ENVS) + "@{}}", "\\toprule",
             "Executor & Arbiter decision & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ENV_NAME[e]}}}" for e in b.ENVS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(b.ENVS))),
             " & & " + " & ".join(f"$t{{=}}{t}$" for _, t in cols) + " \\\\", "\\midrule"]
    for exe, arm in FULL_ARMS:
        for j, label in enumerate(("routed at start", "retired later")):
            lead = f"\\multirow{{2}}{{*}}{{{exe}}}" if j == 0 else ""
            vals = []
            for e, t in cols:
                v = look.get((exe, e, t))
                vals.append("--" if v is None else f"{v[j]:.0f}" + (f"$^{{[{v[2]}]}}$" if v[2] < 8 else ""))
            lines.append(f"{lead} & {label} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "table1_arbiter.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote arbiter_activity.csv ({len(rows)} cells) and {os.path.join(OUT, 'table1_arbiter.tex')}")


if __name__ == "__main__":
    main()
