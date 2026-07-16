#!/usr/bin/env bash
#SBATCH --job-name=cube-download
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_download_%j.log
# OGBench-Cube staging step 1: stream-download + extract in one pass (the
# 46.18GB tar.zst is never stored -- curl | zstd | tar), then a schema
# inspection gate (mirrors the Reacher pattern: verify episodes/columns
# BEFORE any port code is written). PRE-REGISTRATION NOTE: after this lands,
# the FIRST compute on Cube is probe_floor.py (derive tau/k), committed
# before any closed-loop cell -- the derivation rule's graduation test.
# Requires ~150GB free (reacher.h5 deletion, done manually by user).
set -eu
URL=https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst
DEST=/lustre/home/ha676/data/cube
mkdir -p "$DEST"
cd "$DEST"
echo "=== streaming download+extract: $URL -> $DEST ==="
df -h /lustre | tail -1
curl -sL "$URL" | zstd -d --stdout | tar -xf -
echo "=== extracted tree ==="
find "$DEST" -maxdepth 3 -exec ls -lhd {} \;
echo "=== schema inspection ==="
source /lustre/home/ha676/le-wm/.venv/bin/activate
python - <<'EOF'
import glob, h5py
for p in glob.glob("/lustre/home/ha676/data/cube/**/*.h5", recursive=True) + \
         glob.glob("/lustre/home/ha676/data/cube/**/*.hdf5", recursive=True):
    print("=====", p)
    with h5py.File(p, "r") as f:
        def walk(name, obj):
            if hasattr(obj, "shape"):
                print(f"  {name}  {obj.shape}  {obj.dtype}")
        f.visititems(walk)
        for k in ("ep_len", "ep_offset"):
            if k in f:
                print(f"  {k}: n={len(f[k])}, total_frames={int(f[k][:].sum()) if k=='ep_len' else '-'}")
EOF
df -h /lustre | tail -1
echo "=== cube staging complete ==="
