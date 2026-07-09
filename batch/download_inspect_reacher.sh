#!/usr/bin/env bash
set -eu
cd /scratch/u6ko/hasoshu.u6ko/data
mkdir -p reacher
cd reacher

echo "=== downloading reacher.tar.zst (23.8GB) ==="
curl -sL -o reacher.tar.zst https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst
ls -la reacher.tar.zst

echo "=== decompressing ==="
tar --zstd -xf reacher.tar.zst
ls -la

echo "=== downloading checkpoint ==="
mkdir -p /home/u6ko/hasoshu.u6ko/le-wm/encoder_reacher
cd /home/u6ko/hasoshu.u6ko/le-wm/encoder_reacher
curl -sL -o config.json https://huggingface.co/quentinll/lewm-reacher/resolve/main/config.json
curl -sL -o weights.pt https://huggingface.co/quentinll/lewm-reacher/resolve/main/weights.pt
ls -la

echo "=== done, ready for inspection ==="
