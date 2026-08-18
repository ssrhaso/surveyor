#!/usr/bin/env python3
"""Cube start-to-goal displacement under LeWM's Fig. 6 protocol vs ours.

Pre-registered as P-DISP-1 before the run. Offline: reads
cached privileged cube positions, runs no policy and loads no model.

WHY. LeWM Fig. 6 reports Random at 48% on OGBench-Cube; whether that column can
be cited turns on whether 48% measures task difficulty or protocol geometry.
cube_single_expert's expert delivers the cube in ~90 steps then IDLES for ~110,
so a start sampled from the idle phase has a goal 25 steps later at the SAME
position: displacement ~0, already inside the 0.04 m success radius, and doing
nothing succeeds.

THE TWO PROTOCOLS, both read from primary sources rather than inferred:

  (a) LeWM, config/eval/cube.yaml + eval.py (their driver, in this repo):
        goal_offset_steps=25, eval_budget=50, num_eval=50
        start sampled UNIFORMLY over every row with
        step_idx <= ep_len - goal_offset - 1, across all episodes.
  (b) ours, surveyor/envs/cube/eval.py --start final:
        goal_offset=150, budget=300, sample_long =>
        start = ep_len - 1 - 150 (goal is the FINAL frame), episodes
        restricted to ep_len > 151 and to --episode-min 8000.

Displacement is ||block_pos[start + goal_offset] - block_pos[start]||_2 in
metres, from the privileged column, i.e. exactly the quantity the success test
thresholds at 0.04 m.

  python -m surveyor.probes.probe_cube_displacement --h5 <cube.h5> \
      --out Results/cube_displacement.json
"""
import argparse
import json

import numpy as np

CUBE_SUCCESS_THRESH = 0.04  # metres; cube_env._compute_successes
POS_COL = "privileged_block_0_pos"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="cube_single_expert.h5")
    p.add_argument("--lewm-offset", type=int, default=25,
                   help="LeWM config/eval/cube.yaml goal_offset_steps")
    p.add_argument("--ours-offset", type=int, default=150,
                   help="our certified protocol's goal offset")
    p.add_argument("--episode-min", type=int, default=8000,
                   help="our protocol's held-out episode floor")
    p.add_argument("--num-eval", type=int, default=50,
                   help="LeWM num_eval, for the seed-matched draw alongside the population")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=float, default=CUBE_SUCCESS_THRESH)
    p.add_argument("--out", default=None, help="write JSON here")
    return p.parse_args()


def summarise(disp, radius, label):
    """Median/p90 plus the fraction already inside the success radius."""
    disp = np.asarray(disp, dtype=np.float64)
    n = len(disp)
    below = float((disp < radius).mean()) if n else float("nan")
    return {
        "label": label,
        "n_pairs": int(n),
        "median_m": float(np.median(disp)) if n else float("nan"),
        "p90_m": float(np.percentile(disp, 90)) if n else float("nan"),
        "mean_m": float(disp.mean()) if n else float("nan"),
        "frac_below_radius": below,
        "pct_below_radius": round(100.0 * below, 2) if n else float("nan"),
    }


def main():
    args = parse_args()
    import h5py

    with h5py.File(args.h5, "r") as f:
        ep_len = f["ep_len"][:].astype(np.int64)
        ep_off = f["ep_offset"][:].astype(np.int64)
        pos = f[POS_COL][:].astype(np.float64)  # (rows, 3)

    n_eps = len(ep_len)
    print(f"[data] {n_eps} episodes, {len(pos)} rows, "
          f"ep_len min/median/max = {ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}")

    def disp_for(ep_ids, starts, offset):
        """||pos[start+offset] - pos[start]|| per (episode, start) pair."""
        a = ep_off[ep_ids] + np.asarray(starts, dtype=np.int64)
        b = a + offset
        return np.linalg.norm(pos[b] - pos[a], axis=1)

    results = {}

    # ---- (a) LeWM protocol: EVERY valid (episode, start) row they sample from
    # (step_idx <= ep_len - goal_offset - 1), not one per episode. The whole
    # population is characterised; a seed-matched n=50 draw is reported beside
    # it to show the draw is representative.
    off_l = args.lewm_offset
    max_start = ep_len - off_l - 1
    eligible = np.nonzero(max_start >= 0)[0]
    ep_ids_all, starts_all = [], []
    for e in eligible:
        ms = int(max_start[e])
        ep_ids_all.append(np.full(ms + 1, e, dtype=np.int64))
        starts_all.append(np.arange(ms + 1, dtype=np.int64))
    ep_ids_all = np.concatenate(ep_ids_all)
    starts_all = np.concatenate(starts_all)
    d_lewm = disp_for(ep_ids_all, starts_all, off_l)
    results["lewm_population"] = summarise(
        d_lewm, args.radius, f"LeWM Fig.6 protocol (offset {off_l}, uniform start, all valid rows)")

    g = np.random.default_rng(args.seed)
    pick = g.choice(len(d_lewm), size=min(args.num_eval, len(d_lewm)), replace=False)
    results["lewm_draw"] = summarise(
        d_lewm[pick], args.radius,
        f"LeWM protocol, seed-{args.seed} draw of n={args.num_eval} (their num_eval)")

    # ---- (b) our certified protocol: sample_long, start = ep_len-1-offset.
    off_o = args.ours_offset
    valid = ep_len > off_o + 1
    valid[:args.episode_min] = False          # --episode-min 8000 held-out floor
    ours_eps = np.nonzero(valid)[0]
    ours_starts = ep_len[ours_eps] - 1 - off_o
    d_ours = disp_for(ours_eps, ours_starts, off_o)
    results["ours_population"] = summarise(
        d_ours, args.radius,
        f"our certified protocol (offset {off_o}, start=ep_len-1-{off_o}, "
        f"episodes >= {args.episode_min})")

    # ---- the same episodes under our protocol WITHOUT the held-out floor, so the
    # floor cannot be mistaken for the source of any difference.
    valid_nofloor = ep_len > off_o + 1
    nf_eps = np.nonzero(valid_nofloor)[0]
    d_ours_nf = disp_for(nf_eps, ep_len[nf_eps] - 1 - off_o, off_o)
    results["ours_population_no_episode_floor"] = summarise(
        d_ours_nf, args.radius, "our protocol, all episodes (no --episode-min)")

    # ---- readout
    print()
    hdr = f"{'protocol':62s} {'n':>8s} {'med(m)':>8s} {'p90(m)':>8s} {'<0.04m':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for key in ("lewm_population", "lewm_draw", "ours_population",
                "ours_population_no_episode_floor"):
        r = results[key]
        print(f"{r['label'][:62]:62s} {r['n_pairs']:8d} {r['median_m']:8.4f} "
              f"{r['p90_m']:8.4f} {r['pct_below_radius']:7.2f}%")

    frac = results["lewm_population"]["frac_below_radius"]
    print()
    print(f"[P-DISP-1] LeWM-protocol episodes starting already inside the "
          f"{args.radius:g} m success radius: {100*frac:.2f}%")
    # Verdict against the frozen prediction (>=30% confirms, <10% refutes).
    if frac >= 0.30:
        verdict = ("CONFIRMED: a large fraction start inside the radius, so LeWM Fig.6's "
                   "Cube Random entry reflects protocol geometry; that column may be "
                   "cited only with this caveat attached.")
    elif frac < 0.10:
        verdict = ("REFUTED: few episodes start inside the radius, so the 48% Random needs "
                   "a different explanation; report the refutation as-is.")
    else:
        verdict = ("AMBIGUOUS: between the declared 10% and 30% triggers; supports neither "
                   "reading, reported as such.")
    print(f"[P-DISP-1] {verdict}")
    results["prereg"] = {
        "id": "P-DISP-1",
        "confirm_at": 0.30, "refute_below": 0.10,
        "measured_frac_below_radius": frac,
        "verdict": verdict.split(":")[0],
    }
    results["protocols"] = {
        "lewm": {"source": "config/eval/cube.yaml + eval.py (LeWM's own driver)",
                 "goal_offset": off_l, "eval_budget": 50, "num_eval": 50,
                 "start_rule": "uniform over rows with step_idx <= ep_len-offset-1"},
        "ours": {"source": "surveyor/envs/cube/eval.py --start final",
                 "goal_offset": off_o, "eval_budget": 2 * off_o,
                 "episode_min": args.episode_min,
                 "start_rule": "start = ep_len-1-offset (goal is the final frame)"},
        "radius_m": args.radius,
    }

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[json] -> {args.out}")


if __name__ == "__main__":
    main()
