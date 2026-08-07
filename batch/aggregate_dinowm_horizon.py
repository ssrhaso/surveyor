#!/usr/bin/env python3
"""Harvest the DINO-WM second-architecture horizon panel.

Reads plan_outputs/horizon_h<H>_s<seed>/{run_config.json,logs.json} written by
batch/isca/dinowm/run_dinowm_horizon.sbatch.

DISCIPLINE, because DINO-WM already burned us once:
  * TRUNCATION. A cell counts only if its SR trace reached max_iter: the banked
    goal_H=15 pair read 0.88 -> 0.06 but was a 12h-walltime kill after two MPC
    iterations, a scheduler artifact. Truncated cells are never pooled.
  * PROTOCOL AGREEMENT. Cells whose n_evals or max_iter disagree are refused
    rather than silently merged.
  * DIRECTIONAL ONLY. At n_evals=10 the per-cell SE is ~5-6pp. These cells may
    support "flat collapses on a second architecture too" and may NEVER carry a
    paired margin. The banner is printed with every readout so a number cannot
    travel without it.

DINO-WM reports success_rate as a running fraction, one row per MPC iteration,
so a cell's value is the LAST logged row.

  python batch/aggregate_dinowm_horizon.py ~/dino_wm/plan_outputs \
      --csv Results/dinowm_horizon.csv --json Results/dinowm_horizon.json
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DIR_RE = re.compile(r"^horizon_h(?P<h>\d+)_s(?P<seed>\d+)$")


def read_cell(d: Path):
    cfg_p, log_p = d / "run_config.json", d / "logs.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    rows = []
    if log_p.exists():
        rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "mpc/success_rate" in r]
    trace = [r["mpc/success_rate"] for r in rows]
    max_iter = cfg.get("max_iter")
    return {
        "goal_H": cfg.get("goal_H"), "seed": cfg.get("seed"),
        "n_evals": cfg.get("n_evals"), "max_iter": max_iter,
        "logged_steps": len(trace), "sr_trace": trace,
        "sr": trace[-1] if trace else None,
        # The job writes truncated_by_walltime only in its post-run check, so
        # the field is ABSENT exactly when the cell did not complete (still
        # running, or walltime-killed); its presence means python exited normally.
        "finished": "truncated_by_walltime" in cfg,
        # Fewer logged steps than max_iter is NOT by itself truncation: DINO-WM
        # stops once every episode has succeeded, so an easy horizon legitimately
        # exits early at SR 1.0; a naive steps<max_iter test would discard the
        # healthiest cells in the panel.
        "early_exit": len(trace) < (max_iter or 0) and bool(trace) and trace[-1] >= 1.0,
        "short_no_converge": len(trace) < (max_iter or 0) and bool(trace) and trace[-1] < 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_outputs", type=Path)
    ap.add_argument("--expect-seeds", type=int, default=4)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cells = []
    for d in sorted(args.plan_outputs.iterdir()):
        if d.is_dir() and DIR_RE.match(d.name):
            c = read_cell(d)
            c["dir"] = d.name
            cells.append(c)
    if not cells:
        raise SystemExit(f"no horizon_h*_s* dirs under {args.plan_outputs}")

    protos = {(c["n_evals"], c["max_iter"]) for c in cells if c["sr"] is not None}
    if len(protos) > 1:
        raise SystemExit(f"REFUSING to pool: cells disagree on (n_evals, max_iter): {protos}")

    good = defaultdict(list)
    bad = defaultdict(list)
    running = defaultdict(list)
    for c in cells:
        if not c["finished"]:
            running[c["goal_H"]].append(c)
        elif c["sr"] is None:
            bad[c["goal_H"]].append(c)
        else:
            good[c["goal_H"]].append(c)

    n_evals, max_iter = (protos.pop() if protos else (None, None))
    print(f"DINO-WM horizon panel  (their PushT, their protocol; "
          f"n_evals={n_evals}, max_iter={max_iter})")
    print("!! DIRECTIONAL ONLY: n_evals is reduced from the released 50, per-cell "
          "SE ~5-6pp.\n!! Supports 'flat collapses on a second architecture too'; "
          "carries NO paired margin.\n")
    print(f"{'goal_H':>7s} {'seeds':>6s} {'mean SR':>8s} {'min':>6s} {'max':>6s}  per-seed")
    print("-" * 66)
    summary = {}
    for h in sorted(good):
        srs = [c["sr"] for c in good[h]]
        mean = sum(srs) / len(srs)
        n_early = sum(c["early_exit"] for c in good[h])
        n_short = sum(c["short_no_converge"] for c in good[h])
        summary[h] = {"n_seeds": len(srs), "mean_sr": mean,
                      "min_sr": min(srs), "max_sr": max(srs), "per_seed": srs,
                      "n_early_exit_all_solved": n_early,
                      "n_short_without_converging": n_short}
        flag = ""
        if n_early:
            flag += f"  [{n_early} exited early, all solved]"
        if n_short:
            flag += f"  [!! {n_short} stopped short WITHOUT converging - inspect]"
        print(f"{h:7d} {len(srs):6d} {mean:8.3f} {min(srs):6.2f} {max(srs):6.2f}  "
              f"{[round(s,2) for s in srs]}{flag}")

    short = [(h, len(v)) for h, v in good.items() if len(v) < args.expect_seeds]
    if short:
        print("\nINCOMPLETE (usable seeds / expected):")
        for h, got in sorted(short):
            print(f"  goal_H={h}: {got}/{args.expect_seeds}")
    if running:
        print("\nSTILL RUNNING (not finished, not counted either way):")
        for h in sorted(running):
            for c in running[h]:
                print(f"  {c['dir']}: {c['logged_steps']}/{c['max_iter']} iters so far, "
                      f"SR-so-far {c['sr']}")
    if bad:
        print("\nEXCLUDED - finished but truncated by walltime, or no SR logged "
              "(the goal_H=15 failure mode; NOT pooled):")
        for h in sorted(bad):
            for c in bad[h]:
                print(f"  {c['dir']}: logged {c['logged_steps']}/{c['max_iter']} iters, "
                      f"last SR {c['sr']}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["env", "goal_H", "seed", "n_evals", "max_iter",
                        "logged_steps", "truncated", "sr"])
            for c in cells:
                w.writerow(["dinowm_pusht", c["goal_H"], c["seed"], c["n_evals"],
                            c["max_iter"], c["logged_steps"], c["truncated"], c["sr"]])
        print(f"\n[csv] {len(cells)} cells -> {args.csv}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "env": "dinowm_pusht", "protocol": {"n_evals": n_evals, "max_iter": max_iter},
            "directional_only": True,
            "caveat": ("n_evals reduced from the released 50; per-cell SE ~5-6pp; "
                       "no paired margin may be computed from these cells"),
            "by_goal_H": summary,
            "excluded": {str(h): [c["dir"] for c in v] for h, v in bad.items()},
        }, indent=2))
        print(f"[json] -> {args.json}")


if __name__ == "__main__":
    main()
