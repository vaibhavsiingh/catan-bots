"""Summarise strat_stats JSONL files.

usage: python strat_report.py FILE.jsonl [...]
"""

import json
import sys

import numpy as np

RES = ["WHEAT", "ORE", "SHEEP", "WOOD", "BRICK"]


def load(p):
    return [json.loads(l) for l in open(p)]


def summary(games):
    for i, name in ((0, "A"), (1, "B")):
        ps = [g[i] for g in games]
        w = np.array([p["win"] for p in ps])
        se = 1.96 * (w.mean() * (1 - w.mean()) / len(w)) ** 0.5
        m = lambda k: np.mean([p[k] for p in ps])
        op = {r: np.mean([p["opening"][r] for p in ps]) for r in RES}
        print(f"  {name} {ps[0]['bot'][:40]:40s} win {w.mean():.3f}±{se:.3f}  "
              f"open " + " ".join(f"{r[:2]}{op[r]:.1f}" for r in RES)
              + f"  S{m('settlements'):.1f} C{m('cities'):.1f} R{m('roads'):.1f} dev{m('devs'):.1f} "
              f"kn{m('knights'):.1f} army{m('army'):.2f} LR{m('longest'):.2f} trades{m('trades'):.1f}")
    # how winners got their points
    ws = [p for g in games for p in g if p["win"]]
    comp = lambda k: np.mean([p[k] for p in ws])
    print(f"  winners' VP: settle {comp('settlements'):.1f} cities x2 {2*comp('cities'):.1f} "
          f"VPcards {comp('vp_cards'):.1f} army {2*comp('army'):.1f} LR {2*comp('longest'):.1f}  turns {comp('turns'):.0f}")


def logistic(X, y, l2=1e-2, iters=3000, lr=0.1):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w[:-1]


def opening_regression(games):
    """win ~ (my opening pips - opp opening pips) per resource, standardised coefficients."""
    X, y = [], []
    for g in games:
        for i in (0, 1):
            me, op = g[i], g[1 - i]
            X.append([me["opening"][r] - op["opening"][r] for r in RES] + [float(me["first"])])
            y.append(float(me["win"]))
    c = logistic(np.array(X, float), np.array(y))
    print("  opening-pip effect on win (std. logit coef): "
          + " ".join(f"{r}{v:+.2f}" for r, v in zip(RES + ["FIRST"], c)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        g = load(p)
        print(f"{p.split('/')[-1]}  ({len(g)} games)")
        summary(g)
        opening_regression(g)
