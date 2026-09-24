"""Extract the arbiter banners from run logs, for build_arbiter_table.py.

Scans the given directories (recursively, default logs/) for router logs and writes, next to this script,
  arbiter_banners_pulled.txt   <log name>\t<last '[router+surveyor] ... retired=A/N (fired-at-first-replan=B ...' line>
  gcrouter_banners_pulled.txt  <log name>\t<last '[gcidm+surveyor cost] ... routed=R arrived=V ...' line>
in the format of the other banner files (arbiter_banners_A.txt etc.), so the same log found in several directories
resolves to one entry. build_arbiter_table.py reads all of them.

    python reproduce/extract_banners.py logs
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = re.compile(r"^rb[A-Za-z0-9]+_(pusht|reacher|cube|tworoom)_.+_t\d+_seed\d+\.log$")
SEARCH = re.compile(r"retired=\d+/\d+ \(fired-at-first-replan=\d+")
GCIDM = re.compile(r"\[gcidm\+surveyor cost\].*routed=\d+ arrived=\d+")


def main(dirs):
    search, gcidm = {}, {}
    n_files = 0
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.log"), recursive=True):
            name = os.path.basename(f)
            if not NAME.match(name) or not ("router" in name):
                continue
            n_files += 1
            last_s = last_g = None
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if SEARCH.search(line):
                        last_s = line.rstrip("\n")
                    elif GCIDM.search(line):
                        last_g = line.rstrip("\n")
            if last_s:
                search[name] = last_s
            if last_g:
                gcidm[name] = last_g
    for fn, d in (("arbiter_banners_pulled.txt", search), ("gcrouter_banners_pulled.txt", gcidm)):
        with open(os.path.join(HERE, fn), "w", encoding="utf-8", newline="\n") as out:
            for name in sorted(d):
                out.write(f"{name}\t{d[name]}\n")
        print(f"{fn}: {len(d)} logs")
    print(f"scanned {n_files} router logs")


if __name__ == "__main__":
    main(sys.argv[1:] or ["logs"])
