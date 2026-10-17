"""Results table from the game records in a results folder (tools/gauntlet.sh, tools/gauntlet_native.sh).

usage: python tools/summarize.py [results/gauntlet]      -> markdown table on stdout

Each opponent folder holds either harness runs (one sub-folder per arm: distill/, F/) or one native run
(native_<opponent>_<arm>/). Win rates are for our agent; the paired gap compares distill and F board by board
(both seats of a board, same dice), with a 95% interval.
"""

import glob
import json
import math
import os
import sys


def games(folder):
    """every finished game in a folder of chunk files, keyed by (seed0, game index)"""
    out = {}
    for f in glob.glob(os.path.join(folder, "*.json")):
        if os.path.basename(f) != "arm.json":
            for r in json.load(open(f)):
                out[(r.get("seed0", 0), r["game"])] = r
    return out


def rate(g):
    k = len(g)
    w = sum(r["winner"] == "a" for r in g.values())
    p = w / k
    return p, 1.96 * math.sqrt(p * (1 - p) / k), k


def paired(a, b):
    """per board: (wins with a - wins with b) / 2, mean and 95% interval"""
    boards = {}
    for key in a.keys() & b.keys():
        bk = (key[0], key[1] // 2)
        boards.setdefault(bk, []).append((a[key]["winner"] == "a") - (b[key]["winner"] == "a"))
    d = [sum(v) / 2 for v in boards.values() if len(v) == 2]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, 1.96 * sd / math.sqrt(len(d))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gauntlet")
    rows = {}
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            continue
        if d.startswith("native_"):
            opp, arm = d[len("native_"):].rsplit("_", 1)
            rows.setdefault(("native " + opp), {})[arm] = games(path)
        else:
            for arm in os.listdir(path):
                if os.path.isdir(os.path.join(path, arm)):
                    rows.setdefault(d, {})[arm] = games(os.path.join(path, arm))
    print("| opponent | games | distill | F | paired gap (distill - F) |")
    print("|---|---|---|---|---|")
    for opp, arms in rows.items():
        cells = []
        for arm in ("distill", "F"):
            g = arms.get(arm)
            cells.append("" if not g else "{:.1f}% ± {:.1f}".format(100 * rate(g)[0], 100 * rate(g)[1]))
        n = max(len(g) for g in arms.values())
        gap = paired(arms["distill"], arms["F"]) if "distill" in arms and "F" in arms and not opp.startswith("native") else None
        print(f"| {opp} | {n} | {cells[0]} | {cells[1]} | {'' if gap is None else f'{100 * gap[0]:+.1f} ± {100 * gap[1]:.1f}'} |")


if __name__ == "__main__":
    main()
