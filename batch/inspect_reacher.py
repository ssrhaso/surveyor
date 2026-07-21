"""Generic inspection of the decompressed Reacher dataset.

The schema and format are unknown ahead of time (unlike TwoRoom, no prior
exploration was done), so this lists whatever is actually there and adapts
rather than assuming column names.
"""
import glob
import os

DATA_DIR = os.environ.get("REACHER_DATA_DIR", "/scratch/u6ko/hasoshu.u6ko/data/reacher")

print("=== files in", DATA_DIR, "===")
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        p = os.path.join(root, f)
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        print(f"  {p}  ({size/1e9:.2f} GB)" if size > 1e6 else f"  {p}  ({size} B)")

h5_candidates = glob.glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True) + \
    glob.glob(os.path.join(DATA_DIR, "**", "*.hdf5"), recursive=True)
lance_candidates = glob.glob(os.path.join(DATA_DIR, "**", "*.lance"), recursive=True)

print("\nh5 candidates:", h5_candidates)
print("lance candidates:", lance_candidates)

if h5_candidates:
    import hdf5plugin  # noqa: F401
    import h5py
    import numpy as np
    path = h5_candidates[0]
    print(f"\n=== inspecting {path} ===")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print("keys:", keys)
        if "ep_len" in keys:
            ep_len = f["ep_len"][:]
            print("n episodes:", len(ep_len), "mean:", ep_len.mean(),
                  "median:", np.median(ep_len), "min:", ep_len.min(), "max:", ep_len.max())
            maxlen = ep_len.max()
            print(f"frac at max ep_len ({maxlen}):", (ep_len == maxlen).mean())
        for candidate in ["distance_to_target", "reward", "terminated", "truncated", "success"]:
            if candidate in keys:
                print(f"'{candidate}' column present, shape:", f[candidate].shape)
elif lance_candidates:
    print(f"\n=== Lance format found at {lance_candidates[0]}; needs a different reader, "
          f"not inspected here (h5py won't open this) ===")
else:
    print("\n=== no recognized dataset format found; manual inspection needed ===")
