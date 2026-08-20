"""Walk the DINO-WM OSF project (view-only) and list/download files.

  python osf_fetch.py list
  python osf_fetch.py get <substring> <outdir>
"""
import json
import os
import sys
import time
import urllib.request

VIEW = "a56a296ce3b24cceaf408383a175ce28"
BASE = "https://files.us.osf.io/v1/resources/bmw48/providers/osfstorage"


def api(url):
    """GET `url` as JSON, retrying five times with a widening backoff."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"[retry {attempt}] {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def walk(folder_path="/", prefix=""):
    """Walk the OSF storage tree, yielding (relative name, path, size) per file."""
    sep = "" if folder_path.endswith("/") else "/"
    for item in api(f"{BASE}{folder_path}{sep}?view_only={VIEW}")["data"]:
        a = item["attributes"]
        if a["kind"] == "folder":
            yield from walk(a["path"], prefix + a["name"] + "/")
        else:
            yield prefix + a["name"], a["path"], a.get("size") or 0


def fetch(rel, path, size, outdir):
    """Download one file into `outdir`, skipping it when the size already matches."""
    dst = os.path.join(outdir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f"[skip] {rel}", flush=True)
        return
    url = f"{BASE}{path}?view_only={VIEW}"
    print(f"[get ] {rel} ({size / 1e6:.1f} MB)", flush=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
            return
        except Exception as e:
            print(f"[retry {attempt}] {rel}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up on {rel}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "list":
        tot = 0
        for rel, _path, size in walk():
            tot += size
            print(f"{size / 1e6:10.1f} MB  {rel}")
        print(f"TOTAL {tot / 1e9:.2f} GB")
    else:
        sub, outdir = sys.argv[2], sys.argv[3]
        for rel, path, size in walk():
            if sub.lower() in rel.lower():
                fetch(rel, path, size, outdir)
        print("FETCH DONE")
