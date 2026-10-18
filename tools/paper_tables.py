"""The paper's tables from the game records in a results folder (tools/paper_runs.sh).

usage: python tools/paper_tables.py [results]

Writes OUT/tables.md (markdown), OUT/tables.tex (booktabs tables) and OUT/margin_sweep.csv (for a pgfplots
figure), and prints the markdown. Sections whose folders are missing are skipped:

- headline: results/headline (clocked distill vs F; distill and F against AlphaBeta-2)
- round robin: results/roundrobin (ratings, F = 1500, 95% bootstrap interval)
- head to head: results/h2h (ens_distill vs distill)
- ablations: results/ablation/vs_* (each arm's win rate, and its paired gap to the full agent, board by board)
- margin sweep: the judge's override margin, from the ablation arms (inf = no judge, 0.03 = full)
- stronger baselines: results/gauntlet/native_* (opponent on the true game)

Win rates carry 95% normal intervals; paired gaps are per-board differences (both seats, same dice) in
percentage points with 95% intervals.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roundrobin import ratings  # noqa: E402
from summarize import games, paired, rate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPPS = ["F", "AB2", "lookahead"]
ARMS = [  # (folder, label), in table order
    ("full", "full agent"),
    ("rules_off", "all additions off"),
    ("no_judge", "no value-net judge"),
    ("outcome_judge", "judge trained on outcomes only"),
    ("no_knight_rule", "no knight rule"),
    ("no_opening_model", "no opening model"),
    ("no_dev_term", "no development-card term"),
    ("no_robber_rule", "no robber rule"),
    ("no_planner", "no planner (also turns its chains off)"),
    ("no_chains", "no trade/road chains"),
    ("no_win_check", "no win check"),
]
MARGINS = [("margin_000", 0.0), ("margin_001", 0.01), ("full", 0.03), ("margin_005", 0.05), ("margin_010", 0.1),
           ("no_judge", math.inf)]
NATIVE = [("AB3", "AlphaBeta, depth 3"), ("hex_e2g", "HexMachina e2g"), ("hex_e7b", "HexMachina e7b"),
          ("hex_e5bg", "HexMachina e5bg"), ("G", "GreedyPlayouts"), ("M1000", "MCTS, 1000 playouts"),
          ("AB2", "AlphaBeta, depth 2"), ("M", "MCTS, 100 playouts")]


def pct(g):
    p, ci, _ = rate(g)
    return f"{100 * p:.1f} ± {100 * ci:.1f}"


def gap(x):
    return "" if x is None else f"{100 * x[0]:+.1f} ± {100 * x[1]:.1f}"


def tex(s):
    for a, b in (("±", r"$\pm$"), ("%", r"\%"), ("—", "--"), ("–", "--"), ("∞", r"$\infty$"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


class Out:
    def __init__(self):
        self.md, self.tex = [], []

    def table(self, title, head, rows, note=""):
        self.md += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        self.md += ["| " + " | ".join(r) + " |" for r in rows]
        self.md += ([""] + [note] if note else []) + [""]
        self.tex += [f"% {title}", r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule",
                     " & ".join(tex(h) for h in head) + r" \\", r"\midrule"]
        self.tex += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
        self.tex += [r"\bottomrule", r"\end{tabular}", ""]


def headline(res, out):
    rows = []
    d = os.path.join(res, "headline", "vs_F_clocked", "distill")
    if os.path.isdir(d):
        g = games(d)
        dec = sum(r["a"]["decisions"] for r in g.values())
        cpu = sum(r["a"]["cpu_s"] for r in g.values())
        over = sum(r["a"]["over"] is not None for r in g.values())
        rows.append(["distill vs F (clock on)", str(len(g)), pct(g),
                     f"{1000 * cpu / dec:.0f} ms/decision, {over} games over the clock"])
    d = os.path.join(res, "headline", "vs_AB2")
    if os.path.isdir(d):
        a, b = games(os.path.join(d, "distill")), games(os.path.join(d, "F"))
        rows.append(["distill vs AlphaBeta-2", str(len(a)), pct(a), "paired gap to F: " + gap(paired(a, b)) + " pts"])
        rows.append(["F vs AlphaBeta-2", str(len(b)), pct(b), ""])
    if rows:
        out.table("Headline", ["match", "games", "win %", "notes"], rows)


def roundrobin(res, out):
    d = os.path.join(res, "roundrobin")
    if not os.path.isdir(os.path.join(d, "pairs")):
        return
    names = sorted({n for p in os.listdir(os.path.join(d, "pairs")) for n in p.split("__")})
    wins, names, _, point, boot = ratings(d, names, "F")
    order = sorted(names, key=lambda x: -point[x])
    rows = [[a, f"{point[a]:.0f}", f"{boot[a][4]:.0f} – {boot[a][194]:.0f}",
             str(sum(wins[a][b] + wins[b][a] for b in names))] for a in order]
    out.table("Round robin", ["agent", "rating", "95% interval", "games"], rows,
              "Bradley-Terry, Elo scale, F = 1500; every pair 400 games on the same boards, clock off.")
    rows = [[a] + ["—" if a == b else f"{100 * wins[a][b] / max(1, wins[a][b] + wins[b][a]):.0f}" for b in order]
            for a in order]
    out.table("Round robin: win % of the row agent", [""] + order, rows)


def h2h(res, out):
    d = os.path.join(res, "h2h", "ens_distill")
    if os.path.isdir(d):
        g = games(d)
        out.table("Head to head", ["match", "games", "win %"], [["ens_distill vs distill", str(len(g)), pct(g)]],
                  "Both seats of every board, same dice, clock off.")


def ablations(res, out):
    base = os.path.join(res, "ablation")
    opps = [o for o in OPPS if os.path.isdir(os.path.join(base, f"vs_{o}"))]
    if not opps:
        return
    g = {(o, a): games(os.path.join(base, f"vs_{o}", a)) for o in opps for a, _ in ARMS + MARGINS
         if os.path.isdir(os.path.join(base, f"vs_{o}", a))}
    gaps = {k: paired(v, g[(k[0], "full")]) for k, v in g.items() if k[1] != "full" and (k[0], "full") in g}

    def mean_gap(a):
        xs = [gaps.get((o, a)) for o in opps]
        if None in xs:
            return None
        return sum(x[0] for x in xs) / len(xs), math.sqrt(sum(x[1] ** 2 for x in xs)) / len(xs)

    rows = []
    for a, label in ARMS:
        if not any((o, a) in g for o in opps):
            continue
        cells = [pct(g[(o, a)]) if (o, a) in g else "" for o in opps]
        rows.append([label] + cells + ["" if a == "full" else gap(mean_gap(a))])
    out.table("Ablations", ["arm"] + [f"vs {o}" for o in opps] + ["gap to full, mean (pts)"], rows,
              f"Win % against each opponent ({len(g[(opps[0], 'full')])} games each); the last column averages the "
              "paired per-board gaps to the full agent over the opponents. Turning the planner and the win check "
              "off together (no_turn_search) plays exactly the games of no planner.")
    rows, csv = [], ["margin," + ",".join(f"win_{o},ci_{o}" for o in opps) + ",gap,gap_ci"]
    for a, m in MARGINS:
        if not all((o, a) in g for o in opps):
            continue
        mg = None if a == "full" else mean_gap(a)
        rows.append([("∞ (no judge)" if m == math.inf else f"{m:g}") + (" (default)" if a == "full" else "")]
                    + [pct(g[(o, a)]) for o in opps] + ["0" if a == "full" else gap(mg)])
        csv.append(",".join([("inf" if m == math.inf else f"{m:g}")]
                            + [f"{100 * rate(g[(o, a)])[0]:.2f},{100 * rate(g[(o, a)])[1]:.2f}" for o in opps]
                            + ([f"{100 * mg[0]:.2f},{100 * mg[1]:.2f}"] if mg else ["0,0"])))
    out.table("Margin sweep", ["judge margin"] + [f"vs {o}" for o in opps] + ["gap to default (pts)"], rows,
              "The judge replaces the rule-based move when the value net rates its own best move this much higher "
              "in P(win); 0 lets it choose every main-turn move.")
    with open(os.path.join(res, "margin_sweep.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")


def baselines(res, out):
    base = os.path.join(res, "gauntlet")
    rows = []
    for name, label in NATIVE:
        cells = []
        for arm in ("distill", "F"):
            d = os.path.join(base, f"native_{name}_{arm}")
            cells.append((pct(games(d)), len(games(d))) if os.path.isdir(d) else ("", 0))
        if cells[0][1] or cells[1][1]:
            rows.append([label, str(max(c[1] for c in cells)), cells[0][0], cells[1][0]])
    if rows:
        out.table("Stronger baselines", ["opponent", "games", "distill win %", "F win %"], rows,
                  "The opponent plays on the true game (tools/native_match.py), clock off.")


def main():
    res = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results"))
    out = Out()
    for f in (headline, roundrobin, h2h, ablations, baselines):
        f(res, out)
    with open(os.path.join(res, "tables.md"), "w") as fh:
        fh.write("\n".join(out.md))
    with open(os.path.join(res, "tables.tex"), "w") as fh:
        fh.write("% generated by tools/paper_tables.py\n" + "\n".join(out.tex))
    print("\n".join(out.md))


if __name__ == "__main__":
    main()
