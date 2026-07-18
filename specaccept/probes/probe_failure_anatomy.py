"""Failure anatomy from closed-loop --dump-traces files.

Reconstructs each failed episode's latent trajectory from the per-replan
trace and buckets it (first match wins): near_miss, stuck, diverged, or
unreachable_subgoals. Successful episodes are the reference distribution.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept import encoder


def parse_args():
    p = argparse.ArgumentParser(description="failure anatomy from eval traces")
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--record", default="block_5",
                   help="which (score, angle) record to analyze")
    p.add_argument("--h5", required=True)
    p.add_argument("--goal-offset", type=int, default=150)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def per_env_series(trace, n_envs, dim):
    """trace: list of {replan_idx, z_cond, z_next} -> per-env lists of latents."""
    zc = [[] for _ in range(n_envs)]
    zn = [[] for _ in range(n_envs)]
    for step in trace:
        for r, i in enumerate(step["replan_idx"]):
            zc[int(i)].append(step["z_cond"][r])
            zn[int(i)].append(step["z_next"][r])
    return zc, zn


@torch.no_grad()
def main():
    args = parse_args()
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import h5py

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)

    # collect (success, stats) per episode across all trace files
    rows = []
    goal_cache = {}
    for path in args.traces:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        rec = blob["records"][args.record]
        succ = np.asarray(rec["successes"], dtype=bool)
        eps = rec["episodes_idx"]
        starts = rec["start_steps"]
        n_envs = len(succ)
        dim = rec["trace"][0]["z_cond"].shape[1]
        zc, zn = per_env_series(rec["trace"], n_envs, dim)

        # goal latents (cache by (ep, start))
        need = [(int(e), int(s)) for e, s in zip(eps, starts)
                if (int(e), int(s)) not in goal_cache]
        if need:
            with h5py.File(args.h5, "r") as f:
                ep_off = f["ep_offset"][:]
                ep_len = f["ep_len"][:]
                px = f["pixels"]
                rws = []
                for e, s in need:
                    last = int(ep_off[e]) + int(ep_len[e]) - 1
                    rws.append(min(int(ep_off[e]) + s + args.goal_offset, last))
                rws = np.array(rws)
                order = np.argsort(rws, kind="stable")
                lat = encoder.encode_frames(model, px[rws[order]], device=args.device)
                out = torch.empty_like(lat)
                out[order] = lat
            for (key, g) in zip(need, out):
                goal_cache[key] = g

        for i in range(n_envs):
            if not zc[i]:
                continue
            g = goal_cache[(int(eps[i]), int(starts[i]))]
            Zc = torch.stack(zc[i])                    # (T, D) achieved at replans
            Zn = torch.stack(zn[i])                    # (T, D) commanded subgoals
            d_goal = (Zc - g).norm(dim=1)              # distance-to-goal over time
            commanded = (Zn - Zc).norm(dim=1)          # commanded step sizes
            achieved = (Zc[1:] - Zc[:-1]).norm(dim=1)  # achieved step sizes
            ratio = (commanded[:-1] / achieved.clamp_min(1e-6)) if len(achieved) else torch.tensor([1.0])
            T = len(d_goal)
            last3 = d_goal[max(0, T - T // 3):]
            rows.append(dict(
                file=path.split("traces_")[-1], env=i, success=bool(succ[i]),
                d_final=float(d_goal[-1]), d_min=float(d_goal.min()),
                d_first=float(d_goal[0]),
                plateau=float(last3.max() - last3.min()),
                cmd_ach_ratio=float(ratio.median()),
                n_replans=T))

    ok = [r for r in rows if r["success"]]
    bad = [r for r in rows if not r["success"]]
    ok_final = np.array([r["d_final"] for r in ok])
    p75 = float(np.percentile(ok_final, 75))
    p95 = float(np.percentile(ok_final, 95))
    print(f"[anatomy] record={args.record}  episodes: {len(ok)} success / {len(bad)} failed")
    print(f"[reference] success final-dist-to-goal: p50={np.median(ok_final):.2f} "
          f"p75={p75:.2f} p95={p95:.2f}; cmd/ach ratio p50="
          f"{np.median([r['cmd_ach_ratio'] for r in ok]):.2f}")

    buckets = {"near_miss": [], "unreachable_subgoals": [], "diverged": [], "stuck": []}
    for r in bad:
        if r["d_final"] <= p95:
            buckets["near_miss"].append(r)
        elif r["cmd_ach_ratio"] > 3.0:
            buckets["unreachable_subgoals"].append(r)
        elif r["d_final"] > 1.5 * r["d_min"] + 1.0:
            buckets["diverged"].append(r)
        else:
            buckets["stuck"].append(r)

    print("\n[failed episodes]")
    for name, rs in buckets.items():
        print(f"  {name:>22}: {len(rs)}")
        for r in rs:
            print(f"      {r['file']} env={r['env']:>3} d_first={r['d_first']:6.2f} "
                  f"d_min={r['d_min']:5.2f} d_final={r['d_final']:5.2f} "
                  f"plateau={r['plateau']:5.2f} cmd/ach={r['cmd_ach_ratio']:5.2f}")

    print("\n[read-off] near_miss dominant = precision/tolerance issue (finer stride or "
          "longer budget helps); unreachable dominant = the paper's hypothesized mode "
          "(reachability filtering earns a mechanism); diverged/stuck = control failures.")


if __name__ == "__main__":
    main()
