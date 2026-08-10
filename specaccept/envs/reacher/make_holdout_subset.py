"""Build a holdout-only Reacher h5 (episodes >= EP_MIN), INDEX-COMPATIBLE
with the full file.

Per-episode arrays keep their full length-10000 shape, with ep_len[e]=0 for
dropped episodes, which makes them ineligible under every existing sampler
without any flag changes, and ep_offset re-pointed into the subset's row space
for kept episodes. Row-level columns (pixels/qpos/...) hold only the kept
episodes' rows, in episode order. Compression and chunking are mirrored from the
source so readability is identical.

Usage (on the box that has the full file):
  python batch/make_holdout_subset.py --src data/reacher/reacher.h5 \
      --out reacher_holdout.h5 --ep-min 8000
"""
import argparse

try:
    import hdf5plugin  # noqa: F401  (registers dynamic filters if src uses them)
except ImportError:
    pass
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ep-min", type=int, default=8000)
args = p.parse_args()

with h5py.File(args.src, "r") as f, h5py.File(args.out, "w") as g:
    ep_off = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_eps = len(ep_len)
    n_rows_total = f["pixels"].shape[0]
    kept = np.arange(args.ep_min, n_eps)
    kept_rows = int(ep_len[kept].sum())
    print(f"[subset] keeping eps [{args.ep_min}, {n_eps}) = {len(kept)} eps, "
          f"{kept_rows}/{n_rows_total} rows")

    # new per-episode arrays: full length, zeroed below ep_min
    new_len = ep_len.copy()
    new_len[:args.ep_min] = 0
    new_off = np.zeros_like(ep_off)
    run = 0
    for e in kept:
        new_off[e] = run
        run += int(ep_len[e])

    for key in f.keys():
        src = f[key]
        if src.shape and src.shape[0] == n_rows_total:      # row-level column
            shape = (kept_rows,) + src.shape[1:]
            kw = {}
            if src.compression in ("gzip", "lzf", "szip"):
                kw["compression"] = src.compression
                if src.compression_opts is not None:
                    kw["compression_opts"] = src.compression_opts
            elif src.compression is not None:
                # source uses a dynamic plugin filter (h5py says "unknown");
                # re-compress with zstd via hdf5plugin, the same plugin family
                # the eval stack already reads the source file with
                import hdf5plugin as _hp
                kw.update(_hp.Zstd())
            if src.chunks:
                kw["chunks"] = (min(src.chunks[0], new_len[kept].max()),) + src.chunks[1:]
            dst = g.create_dataset(key, shape=shape, dtype=src.dtype, **kw)
            pos = 0
            for i, e in enumerate(kept):
                off, L = int(ep_off[e]), int(ep_len[e])
                dst[pos:pos + L] = src[off:off + L]
                pos += L
                if i % 200 == 0:
                    print(f"  [{key}] {i}/{len(kept)} eps", flush=True)
            print(f"  [{key}] done: {pos} rows")
        elif src.shape and src.shape[0] == n_eps:            # per-episode column
            if key == "ep_len":
                g.create_dataset(key, data=new_len)
            elif key == "ep_offset":
                g.create_dataset(key, data=new_off)
            else:
                g.create_dataset(key, data=src[:])
            print(f"  [{key}] per-episode copied")
        else:
            g.create_dataset(key, data=src[()])
            print(f"  [{key}] scalar/other copied verbatim")

print(f"[subset] wrote {args.out}")
