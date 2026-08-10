"""Floor probe: derive tau and k from offline measurements of the stack.

Three measurements in the accept test's unit, rel(a, b) = ||a-b|| / ||a||:
  A  temporal floor: consecutive-frame distance;
  B  criterion floor, the primary tau basis: distance between frames whose
     ground-truth states are equivalent under the benchmark's own criterion;
  C  per-k sampler dispersion and bias to the k=50 mean, the k basis.
Rejecting below the criterion floor is re-drafting on noise, so tau is the floor
itself. Offline only: no CEM, no closed loop, minutes per env on one GPU.
"""

from __future__ import annotations

import argparse
import json

try:  # some h5 pixel stores are blosc-compressed; this registers the filter
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py
import numpy as np
import torch

from surveyor import encoder
from surveyor.drafter import load_gdm_planner


def parse_args():
    p = argparse.ArgumentParser(description="Floor probe: encoder resolution vs sampler noise, in accept-test units")
    p.add_argument("--env", choices=["pusht", "reacher", "cube"], required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--gdm-ckpt", default=None, help="drafter ckpt for part C (omit to skip)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-anchors", type=int, default=512, help="anchors for parts A+B")
    p.add_argument("--pair-pool", type=int, default=40000, help="candidate frames scanned for criterion partners")
    p.add_argument("--n-cond", type=int, default=128, help="conditioning states for part C")
    p.add_argument("--n-draws", type=int, default=8, help="repeat draws per (condition, k)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 50])
    p.add_argument("--stride", type=int, default=10, help="reference stride for disp(S)")
    p.add_argument("--goal-gap", type=int, default=100, help="goal frame offset for goal-cond drafters")
    p.add_argument("--episode-min", type=int, default=0,
                   help="only use episodes >= this index (Reacher heldout convention: 8000)")
    p.add_argument("--pos-thresh", type=float, default=encoder.POS_THRESH)
    p.add_argument("--qpos-tol", type=float, default=0.05)
    p.add_argument("--cube-tol", type=float, default=0.04,
                   help="cube success threshold (m); the benchmark's own criterion")
    p.add_argument("--tau", type=float, default=0.20, help="the frozen threshold to test against")
    p.add_argument("--verify-space", choices=["lewm", "dino"], default="lewm",
                   help="which encoder the floors are measured through. 'dino' = "
                        "pooled frozen DINOv2 (surveyor.paired.encode_frames_dino), "
                        "the paired verifier's serving space; tau for a paired arm "
                        "MUST come from a floor measured here, never transferred "
                        "from another space. Part C (sampler dispersion) stays "
                        "planner-native and is skipped unless --gdm-ckpt is given.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def rel(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Accept-test units: ||a-b|| / ||a|| (sources.py:310)."""
    return ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)).cpu().numpy()


def dist_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "n": int(len(x))}


def encode_rows(model, pixels, rows, device, batch_size, encode_fn=None):
    """Encode h5 pixel rows (any order, duplicates ok); returns (len(rows), D).

    h5py fancy indexing requires STRICTLY increasing indices: unique-ify,
    encode each row once, scatter back through the inverse map.
    encode_fn defaults to the LeWM path; --verify-space dino passes
    paired.encode_frames_dino so floors are measured in the paired
    verifier's exact serving space."""
    fn = encode_fn or encoder.encode_frames
    rows = np.asarray(rows)
    uniq, inv = np.unique(rows, return_inverse=True)
    chunks = []
    for i in range(0, len(uniq), 512):
        chunks.append(fn(model, pixels[uniq[i:i + 512]],
                         device=device, batch_size=batch_size))
    return torch.cat(chunks, dim=0)[inv]


def load_states(f, env):
    """Ground-truth state array + the equivalence predicate for this env."""
    if env == "pusht":
        st = f["state"][:]

        def equiv(a, b, ang):
            # the benchmark's own tolerance (encoder.eval_state_tol, vectorized):
            # 4-D agent+block position + block angle
            pos = np.linalg.norm(a[:, :4] - b[:, :4], axis=1)
            angd = np.abs(a[:, 4] - b[:, 4])
            return (pos < encoder.POS_THRESH) & (angd < np.deg2rad(ang))
        return st, equiv
    if env == "cube":
        # FULL-SCENE criterion-equivalence. The benchmark scores only the cube
        # (<=0.04 m, cube_env._compute_successes), but the LeWM encoder sees the
        # whole arm, so matching on the cube ALONE pairs visually different frames
        # (cube fixed, arm anywhere) and inflates the measured floor ~10x. We
        # instead require the cube AND the end-effector (the visually dominant,
        # task-relevant arm state) to match, both at the benchmark's 0.04 m scale,
        # so equivalent frames actually look alike; the floor then measures
        # encoder resolution, not task-irrelevant arm variation.
        assert "privileged_block_0_pos" in f and "proprio_effector_pos" in f, \
            f"cube needs block+effector pos (keys: {list(f.keys())})"
        blk = f["privileged_block_0_pos"][:]        # (N,3)
        eff = f["proprio_effector_pos"][:]          # (N,3)
        st = np.concatenate([blk, eff], axis=1)     # (N,6) = [cube | effector]
        print(f"[state] cube: full-scene [block|effector] {st.shape} (both within tol)")

        def equiv(a, b, ang=None, tol=0.04):
            cube_ok = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) <= tol
            eff_ok = np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=1) <= tol
            return cube_ok & eff_ok
        return st, equiv
    # reacher: auto-detect the qpos column
    key = next((k for k in ("qpos", "state", "proprio") if k in f), None)
    assert key is not None, f"no state-like key in h5 (keys: {list(f.keys())})"
    st = f[key][:]
    print(f"[state] reacher: using h5['{key}'] {st.shape}; joints = first 2 dims")

    def equiv(a, b, ang=None, tol=0.05):
        return (np.abs(a[:, :2] - b[:, :2]) < tol).all(axis=1)
    return st, equiv


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.verify_space == "dino":
        from surveyor import paired
        model = paired.load_dinov2(device=args.device)
        EF = paired.encode_frames_dino
        print("[space] floors measured in pooled frozen DINOv2 "
              "(the paired verifier's serving space)")
    else:
        model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                                  local_dir=args.local_dir, swm_src=args.swm_src,
                                  device=args.device)
        EF = None

    out = {"args": vars(args), "env": args.env, "verify_space": args.verify_space}
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_off" if "ep_off" in f else "ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        states, equiv = load_states(f, args.env)
        n_eps = len(ep_off)
        eligible = np.arange(args.episode_min, n_eps)
        print(f"[data] {args.h5}: {n_eps} episodes, {len(eligible)} eligible "
              f"(episode_min={args.episode_min})")

        # ---- anchors: (ep, t) with room for t+stride and the goal gap ----
        eps = rng.choice(eligible, size=args.n_anchors, replace=True)
        margin = max(args.stride, 1) + 1
        ts = np.array([rng.integers(0, max(int(ep_len[e]) - margin, 1)) for e in eps])
        base = ep_off[eps].astype(np.int64) + ts

        # ---- A. temporal floor ----
        z0 = encode_rows(model, pixels, base, args.device, args.batch_size, EF)
        z1 = encode_rows(model, pixels, base + 1, args.device, args.batch_size, EF)
        zS = encode_rows(model, pixels, np.minimum(base + args.stride,
                         (ep_off[eps] + ep_len[eps] - 1).astype(np.int64)),
                         args.device, args.batch_size, EF)
        out["temporal_floor_disp1"] = dist_stats(rel(z1, z0))
        out[f"disp_S{args.stride}"] = dist_stats(rel(zS, z0))
        print(f"[A] temporal floor disp(1): {out['temporal_floor_disp1']}")
        print(f"[A] reference disp(S={args.stride}): {out[f'disp_S{args.stride}']}")

        # ---- B. criterion floor: success-equivalent pairs, different episodes ----
        pool_rows = rng.choice(np.concatenate([np.arange(int(ep_off[e]),
                                                         int(ep_off[e]) + int(ep_len[e]))
                                               for e in eligible]),
                               size=min(args.pair_pool, int(ep_len[eligible].sum())),
                               replace=False)
        pool_states = states[np.sort(pool_rows)]
        pool_rows = np.sort(pool_rows)
        pool_eps = np.searchsorted(ep_off, pool_rows, side="right") - 1
        crit_variants = ([20.0, 5.0] if args.env == "pusht" else [None])
        pair_sets = {}
        for ang in crit_variants:
            pa, pb = [], []
            for i in range(len(base)):
                a_state = np.repeat(states[base[i]][None, :], len(pool_rows), axis=0)
                m = equiv(a_state, pool_states, ang) & (pool_eps != eps[i])
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                pa.append(base[i])
                pb.append(pool_rows[rng.choice(idx)])
            pair_sets[ang] = (np.array(pa), np.array(pb))
            crit_tag = ("%gdeg" % ang) if ang else args.env
            print(f"[B] criterion pairs ({crit_tag}): "
                  f"{len(pa)}/{len(base)} anchors matched")
        for ang, (pa, pb) in pair_sets.items():
            if len(pa) == 0:
                continue
            za = encode_rows(model, pixels, pa, args.device, args.batch_size, EF)
            zb = encode_rows(model, pixels, pb, args.device, args.batch_size, EF)
            key = ("criterion_floor_%gdeg" % ang) if ang else f"criterion_floor_{args.env}"
            out[key] = dist_stats(rel(za, zb))
            print(f"[B] {key}: {out[key]}")

        # ---- C. sampler dispersion per k ----
        if args.gdm_ckpt and args.verify_space == "dino":
            raise ValueError("part C conditions the drafter on the LeWM encode; "
                             "run it with --verify-space lewm (or use the env's "
                             "probe_k for a paired drafter)")
        if args.gdm_ckpt:
            planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
            cond_rows = base[:args.n_cond]
            z_cond = encode_rows(model, pixels, cond_rows, args.device,
                                 args.batch_size).to(planner.device)
            z_goal = None
            if planner.goal_cond:
                goal_rows = np.minimum(cond_rows + args.goal_gap,
                                       (ep_off[eps[:args.n_cond]]
                                        + ep_len[eps[:args.n_cond]] - 1).astype(np.int64))
                z_goal = encode_rows(model, pixels, goal_rows, args.device,
                                     args.batch_size).to(planner.device)
                print(f"[C] goal-conditioned drafter: goals at +{args.goal_gap}")
            disp_by_k, w_mean_by_k = {}, {}
            for k in args.ks:
                gen = torch.Generator(device=planner.device)
                gen.manual_seed(args.seed + k)
                draws = []
                for _ in range(args.n_draws):
                    blk = planner.sample_sequence(z_cond, n_steps=k, generator=gen,
                                                  z_goal_native=z_goal)  # (M, N, D)
                    draws.append(blk[:, 0].detach())        # position-1 waypoint
                W = torch.stack(draws, dim=0)               # (R, M, D)
                w_mean = W.mean(dim=0)                      # (M, D)
                d = ((W - w_mean.unsqueeze(0)).norm(dim=-1)
                     / z_cond.norm(dim=-1).clamp_min(1e-8).unsqueeze(0))
                disp_by_k[k] = dist_stats(d.flatten().cpu().numpy())
                w_mean_by_k[k] = w_mean
                print(f"[C] k={k:>3d} dispersion: {disp_by_k[k]}")
            ref = max(args.ks)
            bias = {k: dist_stats(rel(w_mean_by_k[ref], w_mean_by_k[k]))
                    for k in args.ks if k != ref}
            out["sampler_dispersion_by_k"] = {str(k): v for k, v in disp_by_k.items()}
            out[f"bias_to_k{ref}"] = {str(k): v for k, v in bias.items()}
            for k, v in bias.items():
                print(f"[C] bias k={k} vs k={ref}: p50={v['p50']:.4f}")

    # ---- verdict ----
    print("\n==== verdict ====")
    crit_key = next((k for k in out if k.startswith("criterion_floor")), None)
    if crit_key:
        fl = out[crit_key]
        lo, hi = fl["p50"], 1.3 * fl["p90"]
        print(f"criterion floor ({crit_key}): p50={fl['p50']:.4f} p90={fl['p90']:.4f}")
        print(f"tau*={args.tau:.2f} vs floor band [{lo:.3f}, {hi:.3f}]: "
              f"{'FLOOR-CONSISTENT' if lo <= args.tau <= hi else 'OFF-FLOOR'}")
    if args.gdm_ckpt and crit_key:
        fl50 = out[crit_key]["p50"]
        for k in args.ks:
            d = out["sampler_dispersion_by_k"][str(k)]["p50"]
            print(f"k={k:>3d}: sampler dispersion p50={d:.4f} "
                  f"({'ABOVE' if d > fl50 else 'below'} floor p50) "
                  f"-> {'verifier should reject noise-drafts' if d > fl50 else 'drafts resolvable'}")
        print("interpretation: floor k-independent by construction (B never touches the "
              "drafter); if dispersion crosses the floor between k=2 and k=4, the "
              "k-cliff and the call_ratio->1.00 self-test are the same phenomenon.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
