"""Filmstrip renderer for --dump-strip npz files (qualitative figures).

One row per episode: [start] [replan frames ...] [final] | [goal].
Each replan frame shows the achieved state AT that replan boundary -- exactly
what the verifier saw -- bordered and labeled by the verifier's decision there:

    advance  (green  #0ca30c)  waypoint verified, next block position served
    redraft  (red    #d03b3b)  verification rejected (or no block yet)
    gate     (yellow #fab219)  arrival gate held the goal
    (gray)                     source logs no events (flat / lerp arms)

Color never carries meaning alone: every frame also gets a text label
("r7 acc" / "r7 re" / "r7 goal"), per the status-color rule.

Usage:
    python -m surveyor.render_strips --npz strips/strip_*.npz --out strips/render
    # optional: --episodes 0 3 5   --max-cols 8   --only-success/--only-failure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"advance": "#0ca30c", "redraft": "#d03b3b", "gate": "#fab219",
          "route": "#3b6bd0", "none": "#8a8a8a", "meta": "#555555",
          "goal": "#333333"}
SHORT = {"advance": "acc", "redraft": "re", "gate": "goal", "route": "rt"}


def parse_args():
    p = argparse.ArgumentParser(description="render --dump-strip npz to filmstrips")
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--out", default="strips/render")
    p.add_argument("--max-cols", type=int, default=8,
                   help="max replan frames per row (evenly subsampled, first+last kept)")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="episode (env-slot) indices to render; default all captured")
    p.add_argument("--only-success", action="store_true")
    p.add_argument("--only-failure", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def subsample(n, k):
    if n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(idx.tolist()))


def episode_events(z, i):
    """Ordered event (kind, rel) list for env i; [] when the arm logs none."""
    if "ev_env" not in z:
        return []
    m = z["ev_env"] == i
    kinds = [k if isinstance(k, str) else k.decode() if isinstance(k, bytes) else str(k)
             for k in z["ev_kind"][m]]
    return list(zip(kinds, z["ev_rel"][m].tolist(), strict=True))


def render_row(ax_row, z, i, max_cols):
    frames = z[f"ep{i}_frames"]
    tags = z[f"ep{i}_tags"]
    events = episode_events(z, i)
    # alignment guarantee (see sources.py): k-th strip frame of env i <-> k-th
    # event of env i. If an arm logged no events, borders fall back to gray.
    cells = []
    if f"ep{i}_start" in z:
        cells.append((z[f"ep{i}_start"], "start", "meta"))
    keep = subsample(len(frames), max_cols)
    for j in keep:
        kind = events[j][0] if j < len(events) else "none"
        lab = f"r{int(tags[j])} {SHORT.get(kind, '')}".strip()
        cells.append((frames[j], lab, kind))
    if f"ep{i}_last" in z:
        cells.append((z[f"ep{i}_last"], "final", "meta"))
    if f"ep{i}_goal" in z:
        cells.append((z[f"ep{i}_goal"], "GOAL", "goal"))

    for ax, (img, lab, kind) in zip(ax_row, cells, strict=False):
        ax.imshow(np.asarray(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(COLORS.get(kind, COLORS["none"]))
            s.set_linewidth(3.0)
        ax.set_xlabel(lab, fontsize=7, color="#333333", labelpad=2)
    for ax in ax_row[len(cells):]:
        ax.axis("off")
    return len(cells)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in args.npz:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        succ = z["success"] if "success" in z else None
        eps = args.episodes
        if eps is None:
            eps = sorted(int(k[2:-7]) for k in z.files
                         if k.startswith("ep") and k.endswith("_frames"))
        if args.only_success and succ is not None:
            eps = [i for i in eps if bool(succ[i])]
        if args.only_failure and succ is not None:
            eps = [i for i in eps if not bool(succ[i])]
        if not eps:
            print(f"[skip] {path}: no episodes after filtering")
            continue

        ncols = args.max_cols + 3                      # start + replans + final + goal
        fig, axes = plt.subplots(len(eps), ncols,
                                 figsize=(1.35 * ncols, 1.65 * len(eps)),
                                 squeeze=False)
        for row, i in enumerate(eps):
            render_row(axes[row], z, i, args.max_cols)
            ok = "" if succ is None else ("  success" if succ[i] else "  FAILURE")
            axes[row][0].set_ylabel(f"ep{i}{ok}", fontsize=8, color="#333333")
        arm = meta.get("subgoal", "?")
        env = meta.get("env", "?")
        fig.suptitle(f"{env} / {arm}  t={meta.get('goal_offset', '?')}  "
                     f"seed={meta.get('seed', '?')}  SR={meta.get('sr', float('nan')):.1f}%  "
                     f"(border: green=verified advance, red=redraft, yellow=goal-gate)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        stem = Path(path).stem
        outp = outdir / f"{stem}.png"
        fig.savefig(outp, dpi=args.dpi)
        plt.close(fig)
        print(f"[render] {outp}  ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
