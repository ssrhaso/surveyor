"""Make the cached torch-hub dinov2 checkout py3.9-compatible: current main
uses PEP 604 unions (float | None) in evaluated annotation positions, which
raise TypeError on 3.9. Inserting `from __future__ import annotations` defers
annotation evaluation. Idempotent."""
import pathlib

ROOT = pathlib.Path("/lustre/home/ha676/torch_cache/hub/facebookresearch_dinov2_main")
FUT = "from __future__ import annotations\n"

patched = 0
for p in ROOT.rglob("*.py"):
    src = p.read_text()
    if "from __future__ import annotations" in src:
        continue
    lines = src.splitlines(keepends=True)
    i = 0
    # skip shebang, blank lines, comment header
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    # skip a module docstring if present
    if i < len(lines) and lines[i].lstrip().startswith(('"""', "'''")):
        q = lines[i].lstrip()[:3]
        if lines[i].strip().endswith(q) and len(lines[i].strip()) >= 6:
            i += 1
        else:
            i += 1
            while i < len(lines) and q not in lines[i]:
                i += 1
            i += 1
    lines.insert(i, FUT)
    p.write_text("".join(lines))
    patched += 1
print(f"patched {patched} files under {ROOT}")

import torch  # verify

m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("DINOV2 LOAD OK", sum(x.numel() for x in m.parameters()) / 1e6, "M params")
