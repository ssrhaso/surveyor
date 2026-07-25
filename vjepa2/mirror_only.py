"""Login-node mirror step: compute nodes reach GCS at ~170KB/s, the login
node at ~7MB/s (different route). Run under nohup; writes MIRROR_DONE with
the version dir when complete so the compute job can be gated on it."""
import sys
from pathlib import Path

# import the module FILE directly: going through the specaccept_vjepa2
# package would execute its __init__, which imports torch (absent in the
# TF venv this runs under)
sys.path.insert(0, str(Path(__file__).resolve().parent / "specaccept_vjepa2"))
from fetch_droid import mirror_droid  # noqa: E402

root = Path("/lustre/home/ha676/data/droid_mirror")
vdir = mirror_droid("droid", root, max_shards=60)
(root / "MIRROR_DONE").write_text(str(vdir))
print(f"[done] {vdir}")
