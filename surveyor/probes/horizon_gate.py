"""Horizon-gate probe v1 (offline): is encoder distance rel(z0, z_goal) a
usable episode-level horizon signal?

Gate rule under test: plan flat for the whole episode iff h = rel(z0, z_goal)
<= theta, with theta read from the floor probe's recorded disp_p50(S=10) and
never fitted here. Pre-registered kill condition: fire-rate >= 0.70 at t=25 and
<= 0.15 at t=150, with no amendment of theta to pass. Latents come from the raw
h5 (encoded here) or the frame-identical stride-1 dense dump. Offline only: two
latents per episode-offset pair, no CEM, no closed loop. Result: FAILED as
specified, since rel saturates by t=50.
"""

from __future__ import annotations

import argparse
import json

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import numpy as np
import torch

from surveyor.probes.probe_floor import dist_stats, rel

# pre-registered separation condition: constants, not flags, so they cannot
# be tuned from the command line after seeing the data
T_SHORT, FIRE_MIN_SHORT = 25, 0.70
T_LONG, FIRE_MAX_LONG = 150, 0.15


def parse_args():
    p = argparse.ArgumentParser(description="Episode-level horizon gate, Phase 1: "
                                            "offline h = rel(z0, z_goal) distributions")
    p.add_argument("--h5", default=None, help="raw pixel dataset (encoded here)")
    p.add_argument("--dense-pt", default=None,
                   help="stride-1 latent dump from build_subgoals (pixel-free path)")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--episodes-files", nargs="+", required=True,
                   help="fixed-population JSONs from build_reacher_horizon_episodes.py; "
                        "labeled by their own max_offset field")
    p.add_argument("--offsets", type=int, nargs="+",
                   default=[25, 50, 75, 100, 125, 150])
    p.add_argument("--floor-json", default="Results/floor_reacher.json",
                   help="probe_floor output; theta = disp_S{stride}.p50 is READ from "
                        "here, never recomputed here")
    p.add_argument("--stride", type=int, default=10,
                   help="subgoal stride whose displacement defines theta")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def make_latent_lookup(args):
    """Returns (z_at(eps, steps) -> (n, D) tensor, ep_len_of(eps) -> (n,) array)."""
    if args.dense_pt:
        d = torch.load(args.dense_pt, map_location="cpu")
        assert int(d["stride"]) == 1, f"dense path needs stride-1 (got {d['stride']})"
        lat, offs = d["latents"], d["offsets"].numpy()
        pos = {int(e): i for i, e in enumerate(d["episode_idx"].numpy())}
        ep_len_all = d["ep_len"].numpy()
        print(f"[data] {args.dense_pt}: {len(pos)} episodes, latents {tuple(lat.shape)} "
              f"(frozen encoder, batch-independent -> identical to the pixel path)")

        def z_at(eps, steps):
            k = np.array([pos[int(e)] for e in eps])
            return lat[offs[k] + np.asarray(steps)]

        def ep_len_of(eps):
            return ep_len_all[np.array([pos[int(e)] for e in eps])]
        return z_at, ep_len_of, None

    import h5py
    from surveyor import encoder
    from surveyor.probes.probe_floor import encode_rows
    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    f = h5py.File(args.h5, "r")
    ep_off = f["ep_off" if "ep_off" in f else "ep_offset"][:]
    ep_len_all = f["ep_len"][:]
    pixels = f["pixels"]
    print(f"[data] {args.h5}: {len(ep_off)} episodes (pixel path)")

    def z_at(eps, steps):
        rows = ep_off[np.asarray(eps)] + np.asarray(steps)
        return encode_rows(model, pixels, rows, args.device, args.batch_size)

    def ep_len_of(eps):
        return ep_len_all[np.asarray(eps)]
    return z_at, ep_len_of, f


def main():
    args = parse_args()
    assert (args.h5 is None) != (args.dense_pt is None), \
        "pass exactly one of --h5 / --dense-pt"

    # ---- theta: the already-measured one-subgoal displacement ----
    with open(args.floor_json) as fh:
        floor = json.load(fh)
    key = f"disp_S{args.stride}"
    assert key in floor, f"{args.floor_json} has no '{key}' (keys: {list(floor)})"
    theta = float(floor[key]["p50"])
    print(f"[theta] {theta:.4f} = {key}.p50 from {args.floor_json} "
          f"(derived from probe_floor's measurement; not fitted here)")
    print(f"[pre-registered] fire-rate >= {FIRE_MIN_SHORT} at t={T_SHORT} "
          f"and <= {FIRE_MAX_LONG} at t={T_LONG}, else the gate is dead")

    z_at, ep_len_of, h5_handle = make_latent_lookup(args)

    out = {"args": vars(args), "theta": theta,
           "theta_source": f"{args.floor_json}:{key}.p50",
           "pre_registered": {"t_short": T_SHORT, "fire_min_short": FIRE_MIN_SHORT,
                              "t_long": T_LONG, "fire_max_long": FIRE_MAX_LONG},
           "populations": {}}
    for ep_file in args.episodes_files:
        with open(ep_file) as fh:
            payload = json.load(fh)
        pairs = payload["episodes"]
        label = f"max{payload['max_offset']}"
        eps = np.array([p[0] for p in pairs], dtype=np.int64)
        starts = np.array([p[1] for p in pairs], dtype=np.int64)
        print(f"\n[pop {label}] {ep_file}: {len(pairs)} pairs "
              f"(max_offset={payload['max_offset']}, seed={payload.get('seed')}, "
              f"episode_min={payload.get('episode_min')})")

        z0 = z_at(eps, starts)
        L = ep_len_of(eps)

        cells = {}
        for t in args.offsets:
            goal_steps = starts + t
            valid = goal_steps <= L - 1
            n_drop = int((~valid).sum())
            if valid.sum() == 0:
                print(f"[pop {label}] t={t}: no valid episodes, skipped")
                continue
            vidx = np.flatnonzero(valid)
            z_goal = z_at(eps[vidx], goal_steps[vidx])
            h = rel(z0[vidx], z_goal)  # ||z0 - z_goal|| / ||z0||
            cell = dist_stats(h)
            cell["fire_rate"] = float((h <= theta).mean())
            cell["n_dropped"] = n_drop
            cells[str(t)] = cell
            drop_note = (f"  [{n_drop} dropped: goal past episode end; "
                         f"population-shifted cell]" if n_drop else "")
            print(f"[pop {label}] t={t:>3d}: h p10={cell['p10']:.3f} "
                  f"p50={cell['p50']:.3f} p90={cell['p90']:.3f} "
                  f"fire(h<={theta:.3f})={cell['fire_rate']:.3f} "
                  f"n={cell['n']}{drop_note}")
        out["populations"][label] = {"file": ep_file, "cells": cells}
    if h5_handle is not None:
        h5_handle.close()

    # ---- verdict against the pre-registered condition ----
    print("\n==== verdict ====")
    checks, evaluated_long = [], False
    for label, pop in out["populations"].items():
        cells = pop["cells"]
        short = cells.get(str(T_SHORT))
        if short is not None and short["n_dropped"] == 0:
            ok = short["fire_rate"] >= FIRE_MIN_SHORT
            checks.append(ok)
            print(f"{label} t={T_SHORT}: fire={short['fire_rate']:.3f} "
                  f"(need >= {FIRE_MIN_SHORT}) -> {'PASS' if ok else 'FAIL'}")
        long_cell = cells.get(str(T_LONG))
        if long_cell is not None:
            if long_cell["n_dropped"] == 0:
                ok = long_cell["fire_rate"] <= FIRE_MAX_LONG
                checks.append(ok)
                evaluated_long = True
                print(f"{label} t={T_LONG}: fire={long_cell['fire_rate']:.3f} "
                      f"(need <= {FIRE_MAX_LONG}) -> {'PASS' if ok else 'FAIL'}")
            else:
                print(f"{label} t={T_LONG}: fire={long_cell['fire_rate']:.3f} but "
                      f"{long_cell['n_dropped']} episodes dropped; population-"
                      f"shifted, excluded from the verdict")
        med = [(int(t), c["p50"])
               for t, c in sorted(cells.items(), key=lambda kv: int(kv[0]))]
        mono = all(a[1] <= b[1] for a, b in zip(med, med[1:], strict=False))
        print(f"{label} h p50 across t: "
              + " ".join(f"{t}:{m:.3f}" for t, m in med)
              + f"  (monotone increasing: {mono})")
    passed = bool(checks) and all(checks) and evaluated_long
    out["verdict"] = {"passed": passed, "n_checks": len(checks)}
    verdict_msg = ("SEPARATES: Phase 1 PASS" if passed else
                   "does NOT meet the pre-registered condition: Phase 1 FAIL, "
                   "gate is dead as specified")
    print(f"\ngate {verdict_msg}")
    print("(theta is derived; the condition is not re-tuned either way)")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
