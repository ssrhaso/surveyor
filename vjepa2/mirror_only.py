"""Login-node mirror step: compute nodes reach GCS at ~170KB/s, the login node
at ~7MB/s over a different route. Run under nohup. Writes MIRROR_DONE with the
version dir on completion, so the compute job can be gated on it."""
import sys
from pathlib import Path

# import the module FILE directly: the specaccept_vjepa2 package's __init__
# imports torch, which is absent in the TF venv this runs under
sys.path.insert(0, str(Path(__file__).resolve().parent / "specaccept_vjepa2"))
from fetch_droid import mirror_droid  # noqa: E402

root = Path("/lustre/home/ha676/data/droid_mirror")
vdir = mirror_droid("droid", root, max_shards=60)
(root / "MIRROR_DONE").write_text(str(vdir))
print(f"[done] {vdir}")
