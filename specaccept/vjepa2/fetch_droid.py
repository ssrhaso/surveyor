"""Fetch DROID episodes (RLDS droid_100 sample) -> probe-format npz.

Reads the public TFDS build at gs://gresearch/robotics (droid_100: 100
episodes, ~2GB) and writes one npz per episode with EXACTLY the keys
probe_offline.py already consumes:
  observations (1, T, H, W, 3) uint8   exterior_image_1_left (upstream trains
                                       on external cams, never wrist)
  states       (1, T, 7)  float32      cartesian_position (xyz + euler xyz)
                                       + gripper_position; the notebook's
                                       compute_new_pose convention, base frame
                                       (camera_frame=False, upstream default)
Frames are subsampled 15Hz -> 5Hz (--subsample 3) to match upstream's
training fps (app/vjepa_droid/droid.py: fps=5).

Needs tensorflow(-cpu) + tensorflow-datasets (its own venv; NOT the le-wm
venv). CPU-only, network-heavy.

  python fetch_droid.py --out-dir /path/droid_eps --num-episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mirror_droid(dataset: str, mirror_root: Path, max_shards: int | None = None) -> Path:
    """Mirror gs://gresearch/robotics/<dataset>/ locally over plain HTTPS
    (GCS JSON listing + object GETs via requests, which ships its own CA
    bundle) so TF never touches the network - its GCS reader needs a Debian
    CA path that HPC nodes often lack. Returns the local version dir
    (the one containing dataset_info.json). Idempotent by size check.
    max_shards: mirror only the first K tfrecord shards (full droid is
    ~1.7TB / ~2048 shards / ~37 eps per shard) plus all metadata files;
    the reader must then use the raw-TFRecord path, not as_dataset."""
    import requests

    prefix = f"robotics/{dataset}/"
    list_url = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
    params = {"prefix": prefix, "maxResults": 1000}
    items = []
    while True:
        r = requests.get(list_url, params=params, timeout=120)
        r.raise_for_status()
        j = r.json()
        items += j.get("items", [])
        if "nextPageToken" not in j:
            break
        params["pageToken"] = j["nextPageToken"]
    assert items, f"no objects listed under gs://gresearch/{prefix}"
    if max_shards is not None:
        shards = sorted(it["name"] for it in items if ".tfrecord" in it["name"])
        keep = set(shards[:max_shards])
        items = [it for it in items
                 if ".tfrecord" not in it["name"] or it["name"] in keep]
        print(f"[mirror] partial: {len(keep)}/{len(shards)} shards + metadata")
    total = sum(int(it["size"]) for it in items)
    print(f"[mirror] {len(items)} objects, {total / 1e9:.2f} GB -> {mirror_root}")
    for it in items:
        rel = it["name"][len("robotics/"):]
        dst = mirror_root / rel
        if dst.exists() and dst.stat().st_size == int(it["size"]):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(f"https://storage.googleapis.com/gresearch/{it['name']}",
                          stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in resp.iter_content(1 << 22):
                    f.write(chunk)
        print(f"[mirror] {rel} ({int(it['size']) / 1e6:.1f} MB)")
    vdirs = [p.parent for p in (mirror_root / dataset).rglob("dataset_info.json")]
    assert vdirs, f"no dataset_info.json under {mirror_root / dataset}"
    # full droid ships metadata for several versions (1.0.0, 1.0.1, ...) but a
    # partial mirror only has shards for one; prefer a version dir that
    # actually contains tfrecords
    with_shards = [v for v in vdirs if any(v.glob("*.tfrecord*"))]
    return sorted(with_shards or vdirs)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--dataset", default="droid_100")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="mirror/read only the first K tfrecord shards (use for the "
                         "full 'droid' dataset: ~37 eps/shard, ~0.8GB/shard)")
    ap.add_argument("--mirror-dir", default=None,
                    help="local mirror root (default: <out-dir>/../droid_100_mirror)")
    ap.add_argument("--camera", default="exterior_image_1_left")
    ap.add_argument("--subsample", type=int, default=3, help="15Hz -> 5Hz")
    ap.add_argument("--min-frames", type=int, default=20,
                    help="skip episodes shorter than this AFTER subsampling")
    args = ap.parse_args()

    import tensorflow_datasets as tfds

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mirror_root = Path(args.mirror_dir) if args.mirror_dir \
        else out.parent / f"{args.dataset}_mirror"
    version_dir = mirror_droid(args.dataset, mirror_root, max_shards=args.max_shards)
    print(f"[mirror] reading local build at {version_dir}")

    builder = tfds.builder_from_directory(str(version_dir))
    if args.max_shards is None:
        ds = builder.as_dataset(split="train")
    else:
        # partial mirror: split metadata references absent shards, so read the
        # present tfrecords directly and decode with the dataset's own features
        import tensorflow as tf
        shard_files = sorted(str(p) for p in version_dir.glob("*.tfrecord*"))
        print(f"[mirror] raw-TFRecord read over {len(shard_files)} shards")
        ds = tf.data.TFRecordDataset(shard_files, num_parallel_reads=4).map(
            builder.info.features.deserialize_example)
    manifest = []
    kept = 0
    for ei, episode in enumerate(ds):
        if kept >= args.num_episodes:
            break
        frames, states = [], []
        lang = ""
        for si, step in enumerate(episode["steps"]):
            if si % args.subsample:
                continue
            obs = step["observation"]
            frames.append(obs[args.camera].numpy())
            cart = obs["cartesian_position"].numpy().astype(np.float32)   # (6,)
            grip = obs["gripper_position"].numpy().astype(np.float32).reshape(-1)[:1]
            states.append(np.concatenate([cart, grip]))
            if not lang and "language_instruction" in step:
                lang = step["language_instruction"].numpy().decode("utf-8", "ignore")
        if len(frames) < args.min_frames:
            print(f"[skip] episode {ei}: only {len(frames)} frames after subsample")
            continue
        obs_arr = np.stack(frames)[None]                    # (1, T, H, W, 3)
        st_arr = np.stack(states)[None].astype(np.float32)  # (1, T, 7)
        name = f"droid_ep{kept:03d}.npz"
        np.savez_compressed(out / name, observations=obs_arr, states=st_arr)
        manifest.append({"file": name, "src_episode": ei, "frames": int(obs_arr.shape[1]),
                         "hw": list(obs_arr.shape[2:4]), "lang": lang})
        kept += 1
        if kept % 10 == 0:
            print(f"[fetch] {kept} episodes written (latest {name}, "
                  f"T={obs_arr.shape[1]}, {lang!r})")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {kept} episodes -> {out} (manifest.json written)")


if __name__ == "__main__":
    main()
