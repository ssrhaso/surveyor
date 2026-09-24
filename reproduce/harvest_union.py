"""Harvest a battery whose logs are split across two clusters.

Both clusters bank the same log names (logs/rb<X>_<env>_<arm>_t<t>_seed<s>.log), so a battery that ran partly on each is
harvested by pointing harvest_rbA.py at one directory that holds the union. This script builds that union in a scratch
directory (a log present in both sources is taken from the first source that has it), runs harvest_rbA.py there for
every requested prefix, and copies the resulting rb<X>_cells.csv / rb<X>_grid.md into reproduce/.

    python reproduce/harvest_union.py --prefix rbL rbM rbN --src logs_A logs_B

Each --src is a directory that contains either the logs directly or one sub-directory per prefix (<src>/rbN/rbN_*.log).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST = os.path.join(HERE, "harvest_rbA.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--keep", default=None, help="keep the union directory here instead of a temp dir")
    args = ap.parse_args()
    work = args.keep or tempfile.mkdtemp(prefix="harvest_union_")
    logs = os.path.join(work, "logs")
    os.makedirs(logs, exist_ok=True)
    for pref in args.prefix:
        n_new, n_dup = 0, 0
        for src in args.src:
            found = glob.glob(os.path.join(src, f"{pref}_*_t*_seed*.log")) + \
                glob.glob(os.path.join(src, pref, f"{pref}_*_t*_seed*.log"))
            for f in found:
                dst = os.path.join(logs, os.path.basename(f))
                if os.path.exists(dst):
                    n_dup += 1
                    continue
                shutil.copy(f, dst)
                n_new += 1
        print(f"{pref}: {n_new} logs in the union, {n_dup} duplicates skipped")
        r = subprocess.run([sys.executable, HARVEST, pref], cwd=work, capture_output=True, text=True,
                           env={**os.environ, "HARVEST_OUT": os.path.join(work, "docs")})
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"harvest failed for {pref}")
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)")
        for name in (f"{pref}_cells.csv", f"{pref}_grid.md"):
            src = os.path.join(work, "docs", name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(HERE, name))
                print(f"  -> reproduce/{name}")
    print(f"union directory: {work}")


if __name__ == "__main__":
    main()
