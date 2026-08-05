"""Aggregate the Reacher eval logs into writeup-ready tables.

Parses logs/reacher_arms_*.log and logs/reacher_horizon_*.log (the tee'd
per-run logs from run_reacher_local.sh) for:
  * the summary line:      "subgoal=... goal_offset=... budget=... seed=..."
  * the SR line:           "  SR = 84.38%  (27/32)"
  * spec-accept telemetry: "[specaccept] ... call_ratio=0.532 ..."

Prints (a) the arms table, per-config SR mean +/- sd over seeds with mean
call_ratio for the spec arms, and (b) the horizon table, arm x offset for both
budget regimes. Run from the repo root on the box that ran the stage:
  python batch/aggregate_reacher_results.py
"""
import glob
import os
import re
import statistics
from collections import defaultdict

SR_RE = re.compile(r"SR = ([\d.]+)%\s+\((\d+)/(\d+)\)")
HDR_RE = re.compile(
    r"subgoal=(\S+) num_eval=(\d+) goal_offset=(\d+) budget=(\d+) seed=(\d+)")
CR_RE = re.compile(r"call_ratio=([\d.]+)")


def parse_log(path):
    """A horizon log can hold TWO runs (2t regime + b50 regime appended);
    return a list of {subgoal, goal_offset, budget, seed, sr, call_ratio}."""
    text = open(path, errors="replace").read()
    runs = []
    hdrs = list(HDR_RE.finditer(text))
    srs = list(SR_RE.finditer(text))
    crs = list(CR_RE.finditer(text))
    for i, (h, s) in enumerate(zip(hdrs, srs)):
        rec = {"subgoal": h.group(1), "num_eval": int(h.group(2)),
               "goal_offset": int(h.group(3)), "budget": int(h.group(4)),
               "seed": int(h.group(5)), "sr": float(s.group(1)),
               "call_ratio": None}
        if i < len(crs):
            rec["call_ratio"] = float(crs[i].group(1))
        runs.append(rec)
    return runs


def fmt_cell(srs, crs):
    if not srs:
        return "--"
    mean = statistics.mean(srs)
    sd = statistics.stdev(srs) if len(srs) > 1 else 0.0
    cell = f"{mean:6.2f} +-{sd:4.2f} (n={len(srs)})"
    crs = [c for c in crs if c is not None]
    if crs:
        cell += f"  cr={statistics.mean(crs):.3f}"
    return cell


def main():
    # ---- arms battery: name from filename reacher_arms_<name>_seed<seed>.log
    arms = defaultdict(lambda: ([], []))  # name -> (srs, call_ratios)
    for path in sorted(glob.glob("logs/reacher_arms_*_seed*.log")):
        m = re.match(r"reacher_arms_(.+)_seed(\d+)\.log", os.path.basename(path))
        if not m:
            continue
        for rec in parse_log(path):
            arms[m.group(1)][0].append(rec["sr"])
            arms[m.group(1)][1].append(rec["call_ratio"])
    if arms:
        print("==== ARMS BATTERY (native protocol: offset 25, budget 50) ====")
        order = ["baseline", "s25gdm", "s15gdm", "s10gdm", "s5gdm",
                 "s25spec", "s15spec", "s10spec", "s5spec"]
        for name in order + sorted(set(arms) - set(order)):
            if name in arms:
                print(f"  {name:10s} {fmt_cell(*arms[name])}")
        print()

    # ---- horizon sweep: reacher_horizon_<name>_t<off>_seed<seed>.log
    hz = defaultdict(lambda: ([], []))  # (name, offset, regime) -> (srs, crs)
    for path in sorted(glob.glob("logs/reacher_horizon_*_t*_seed*.log")):
        m = re.match(r"reacher_horizon_(.+)_t(\d+)_seed(\d+)\.log",
                     os.path.basename(path))
        if not m:
            continue
        name, off = m.group(1), int(m.group(2))
        for rec in parse_log(path):
            regime = "2t" if rec["budget"] == 2 * rec["goal_offset"] else "b50"
            if rec["goal_offset"] == 25 and rec["budget"] == 50:
                regime = "2t"  # coincide at t=25 (the PushT dagger note)
            hz[(name, off, regime)][0].append(rec["sr"])
            hz[(name, off, regime)][1].append(rec["call_ratio"])
    if hz:
        arms_seen = sorted({k[0] for k in hz})
        offs = sorted({k[1] for k in hz})
        for regime, label in [("2t", "FF-JEPA regime (budget = 2t)"),
                              ("b50", "VLWM regime (budget = 50 fixed)")]:
            rows = [(n, o) for n in arms_seen for o in offs
                    if (n, o, regime) in hz]
            if not rows:
                continue
            print(f"==== HORIZON SWEEP: {label} ====")
            for n in arms_seen:
                cells = []
                for o in offs:
                    key = (n, o, regime)
                    cells.append(f"t{o}: {fmt_cell(*hz[key])}" if key in hz
                                 else f"t{o}: --")
                print(f"  {n:10s} " + " | ".join(cells))
            print()


if __name__ == "__main__":
    main()
