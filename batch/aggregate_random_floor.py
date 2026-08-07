#!/usr/bin/env python3
"""Harvest the random-action floor (batch/isca/run_random_baseline.sbatch).

Reads the per-cell logs written by that array and pools successes over seeds
the way every other arm in the paper is pooled: sum the per-seed numerators and
denominators, never average the percentages (cells differ in n).

Two log dialects, both emitted by the drivers as-is:
  PushT   [RESULT] score=block angle=20deg  SR=99.60%  (255/256)
  others    SR = 91.11%  (117/128)
PushT scores at two angles; the paper's grid quotes 20 degrees, so that is what
is pooled, with 5 degrees carried alongside for the appendix.

  python batch/aggregate_random_floor.py logs/            # dir of *.log
  python batch/aggregate_random_floor.py logs/ --csv random_floor.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# randfloor_<env>_t<off>_seed<seed>.log, and cube's offset-free name
NAME_RE = re.compile(
    r"randfloor_(?P<env>pusht|reacher|cube|tworoom)"
    r"(?:_t(?P<off>\d+))?_seed(?P<seed>\d+)\.log$"
)
PUSHT_RE = re.compile(
    r"\[RESULT\].*?angle=(?P<angle>[\d.]+)deg\s+SR=(?P<sr>[\d.]+)%\s+"
    r"\((?P<succ>\d+)/(?P<n>\d+)\)"
)
# MULTILINE matters: this line sits mid-log for cube/reacher/tworoom, so a
# start-of-string anchor silently drops those three envs.
PLAIN_RE = re.compile(r"^[ \t]*SR = (?P<sr>[\d.]+)%\s+\((?P<succ>\d+)/(?P<n>\d+)\)",
                      re.MULTILINE)


def parse_log(path):
    """Yield (angle_or_None, successes, n) for every scored readout in a log."""
    text = path.read_text(errors="replace")
    hits = [(m.group("angle"), int(m.group("succ")), int(m.group("n")))
            for m in PUSHT_RE.finditer(text)]
    if hits:
        return hits
    return [(None, int(m.group("succ")), int(m.group("n")))
            for m in PLAIN_RE.finditer(text)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir", type=Path, help="directory holding randfloor_*.log")
    ap.add_argument("--csv", type=Path, default=None, help="also write per-seed rows here")
    args = ap.parse_args()

    # (env, offset, angle) -> [successes, n, {seeds}]
    pooled = defaultdict(lambda: [0, 0, set()])
    rows = []
    logs = sorted(args.logdir.glob("randfloor_*.log"))
    if not logs:
        raise SystemExit(f"no randfloor_*.log under {args.logdir}")

    for path in logs:
        m = NAME_RE.search(path.name)
        if not m:
            continue
        env = m.group("env")
        off = int(m.group("off")) if m.group("off") else 150  # cube's only cell
        seed = int(m.group("seed"))
        for angle, succ, n in parse_log(path):
            key = (env, off, angle)
            cell = pooled[key]
            # a re-run of the same seed must not be counted twice
            if seed in cell[2]:
                continue
            cell[0] += succ
            cell[1] += n
            cell[2].add(seed)
            rows.append({"env": env, "goal_offset": off, "seed": seed,
                         "angle": angle or "", "successes": succ, "n": n,
                         "sr": round(100.0 * succ / n, 2) if n else ""})

    order = {"pusht": 0, "reacher": 1, "cube": 2, "tworoom": 3}
    print(f"{'env':9s} {'t':>4s} {'angle':>6s} {'seeds':>6s} {'succ/n':>10s} {'SR%':>7s}")
    print("-" * 48)
    for (env, off, angle), (succ, n, seeds) in sorted(
            pooled.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2] or "")):
        sr = 100.0 * succ / n if n else float("nan")
        print(f"{env:9s} {off:4d} {angle or '-':>6s} {len(seeds):6d} "
              f"{f'{succ}/{n}':>10s} {sr:7.2f}")

    incomplete = [(k, len(v[2])) for k, v in pooled.items()]
    expect = {"pusht": 4, "reacher": 8, "cube": 8, "tworoom": 12}
    short = [(k, got) for k, got in incomplete if got < expect[k[0]]]
    if short:
        print("\nINCOMPLETE (cell: seeds present / expected):")
        for (env, off, angle), got in short:
            print(f"  {env} t={off} angle={angle or '-'}: {got}/{expect[env]}")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {len(rows)} per-seed rows -> {args.csv}")


if __name__ == "__main__":
    main()
