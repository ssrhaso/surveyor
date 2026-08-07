#!/usr/bin/env python3
"""Harvest GC-IDM cells (anchor and grid) from the shared eval-driver logs.

Same discipline as batch/aggregate_random_floor.py: pool numerator and
denominator, never average percentages; flag any cell short of its declared
seeds; emit a flat CSV keyed by env / horizon / arm / seed.

Log names are gcidm_<phase>_<env>[_t<offset>]_seed<seed>.log where phase is
"anchor" (LeWM's protocol, goal offset 25 / budget 50 -- the cells that must
reproduce the published numbers before anything is swept) or "grid" (our
populations at budget 2t).

Two dialects, as emitted by the drivers:
  PushT     [RESULT] score=block angle=20deg  SR=99.60%  (255/256)
  others      SR = 91.11%  (117/128)

The call-cost column is carried through because GC-IDM's whole claim is cheap
decisions: the driver prints "[gcidm-cost] decisions=<n>" when available, and one
decision is a single MLP forward against CEM's full solve.

  python batch/aggregate_gcidm.py logs/ --csv Results/gcidm.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# v1 logs are gcidm_<phase>_...; the H_max-corrected rerun writes gcidm2_grid_...
# v1 served one H_max=50 checkpoint at budgets up to 300 (breaking the paper's
# H_max = T rule); v2 trains per cell at H_max = T and supersedes it, but v1
# stays visible so the size of the correction is on the record.
NAME_RE = re.compile(
    r"gcidm(?P<ver>2?)_(?P<phase>anchor|grid)_(?P<env>pusht|reacher|cube|tworoom)"
    r"(?:_t(?P<off>\d+))?_seed(?P<seed>\d+)\.log$")
PUSHT_RE = re.compile(
    r"\[RESULT\].*?angle=(?P<angle>[\d.]+)deg\s+SR=(?P<sr>[\d.]+)%\s+"
    r"\((?P<succ>\d+)/(?P<n>\d+)\)")
# MULTILINE: this line sits mid-log for cube/reacher/tworoom.
PLAIN_RE = re.compile(r"^[ \t]*SR = (?P<sr>[\d.]+)%\s+\((?P<succ>\d+)/(?P<n>\d+)\)",
                      re.MULTILINE)
COST_RE = re.compile(r"\[gcidm-cost\][^\n]*decisions=(?P<d>\d+)")

# expected seeds per (phase, env); anchor mirrors the paper's 3 seeds
EXPECT = {("anchor", "pusht"): 3, ("anchor", "reacher"): 3, ("anchor", "cube"): 3,
          ("anchor", "tworoom"): 3,
          ("grid", "pusht"): 4, ("grid", "reacher"): 8, ("grid", "cube"): 8,
          # v2 runs ONE seed where the arm is deterministic (proved by v1's
          # byte-identical replicas) and keeps 8 on cube, whose population is
          # resampled per seed.
          ("grid2", "pusht"): 1, ("grid2", "reacher"): 1, ("grid2", "cube"): 8}


def parse_log(path):
    text = path.read_text(errors="replace")
    cost = COST_RE.search(text)
    decisions = int(cost.group("d")) if cost else None
    hits = [(m.group("angle"), int(m.group("succ")), int(m.group("n")))
            for m in PUSHT_RE.finditer(text)]
    if not hits:
        hits = [(None, int(m.group("succ")), int(m.group("n")))
                for m in PLAIN_RE.finditer(text)]
    return hits, decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    pooled = defaultdict(lambda: [0, 0, set(), [], [], 0])
    rows = []
    logs = sorted(set(args.logdir.glob("gcidm_*.log"))
                  | set(args.logdir.glob("gcidm2_*.log")))
    if not logs:
        raise SystemExit(f"no gcidm_*.log or gcidm2_*.log under {args.logdir}")

    for path in logs:
        m = NAME_RE.search(path.name)
        if not m:
            continue
        phase = ("grid2" if m.group("ver") else m.group("phase"))
        env = m.group("env")
        off = int(m.group("off")) if m.group("off") else (25 if phase == "anchor" else 150)
        seed = int(m.group("seed"))
        hits, decisions = parse_log(path)
        for angle, succ, n in hits:
            key = (phase, env, off, angle)
            cell = pooled[key]
            if seed in cell[2]:          # a re-run must not double-count
                continue
            cell[0] += succ
            cell[1] += n
            cell[2].add(seed)
            if decisions is not None:
                cell[3].append(decisions)
            cell[4].append(succ)     # per-seed successes, for the replica guard
            cell[5] = n              # episodes per seed (constant within a cell)
            rows.append({"phase": phase, "env": env, "arm": "gcidm",
                         "goal_offset": off, "seed": seed, "angle": angle or "",
                         "successes": succ, "n": n,
                         "sr": round(100.0 * succ / n, 2) if n else "",
                         "decisions": decisions if decisions is not None else ""})

    # ---- DUPLICATE-REPLICA GUARD -------------------------------------------
    # GC-IDM is a deterministic MLP: where the episode file pins the starts and
    # the dynamics are deterministic (Reacher, largely PushT), every seed
    # reproduces the SAME rollouts and pooling seeds overstates the sample.
    # Detect identical per-seed success counts and report n_eff, never the sum.
    def _flagged(key):
        succ = pooled[key][4]
        distinct = len(set(succ))
        if distinct == 1 and len(succ) > 1:
            return "DETERMINISTIC: all seeds identical, n_eff = one draw"
        if distinct <= max(2, len(succ) // 4) and len(succ) > 2:
            return f"near-deterministic: {distinct} distinct values over {len(succ)} seeds"
        return ""

    def replica_stats(key):
        """n_eff and a note. A cell's angles are the SAME rollouts scored at two
        criteria, so if any angle looks deterministic the whole cell is: a looser
        criterion can add distinct values without adding a single new episode."""
        note = _flagged(key)
        if not note:
            siblings = [k for k in pooled
                        if k[0] == key[0] and k[1] == key[1] and k[2] == key[2]]
            for s in siblings:
                sib = _flagged(s)
                if sib:
                    note = sib + " (inherited: same rollouts, other criterion)"
                    break
        return (pooled[key][5], note) if note else (None, "")

    order = {"pusht": 0, "reacher": 1, "cube": 2, "tworoom": 3}
    for phase in ("anchor", "grid", "grid2"):
        keys = [k for k in pooled if k[0] == phase]
        if not keys:
            continue
        print(f"\n=== GC-IDM: {phase} ===")
        print(f"{'env':9s} {'t':>4s} {'angle':>6s} {'seeds':>6s} {'succ/n':>11s} "
              f"{'SR%':>7s} {'n_eff':>7s} {'dec/ep':>7s}  note")
        print("-" * 92)
        notes = []
        for key in sorted(keys, key=lambda k: (order[k[1]], k[2], k[3] or "")):
            succ, n, seeds, dec = pooled[key][:4]
            sr = 100.0 * succ / n if n else float("nan")
            dstr = f"{sum(dec)/len(dec):.0f}" if dec else "-"
            n_eff, note = replica_stats(key)
            eff = n_eff if n_eff is not None else n
            print(f"{key[1]:9s} {key[2]:4d} {key[3] or '-':>6s} {len(seeds):6d} "
                  f"{f'{succ}/{n}':>11s} {sr:7.2f} {eff:7d} {dstr:>7s}  {note}")
            if note:
                notes.append((key, note))
        if notes:
            print("\n!! POOLED n IS NOT THE EFFECTIVE SAMPLE for the cells flagged above.")
            print("!! Quote n_eff, and size error bars from n_eff, never from succ/n.")
        short = [(k, len(pooled[k][2])) for k in keys
                 if len(pooled[k][2]) < EXPECT.get((k[0], k[1]), 0)]
        if short:
            print("INCOMPLETE (seeds present / expected):")
            for k, got in short:
                print(f"  {k[1]} t={k[2]} angle={k[3] or '-'}: "
                      f"{got}/{EXPECT.get((k[0], k[1]))}")

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {len(rows)} per-seed rows -> {args.csv}")


if __name__ == "__main__":
    main()
