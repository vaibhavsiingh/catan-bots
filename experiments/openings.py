"""How much do start positions (the setup phase) decide a 2-player game?

B  replay:  play the setup once (F vs F), then replay the rest of the game M times with different
            dice / dev-deck order from that exact position -> P(first player wins | opening).
C  forced:  same, but the first player's FIRST settlement is forced to F's rank-r corner (r=0..4),
            the rest of the setup is F's choice -> what picking a worse corner costs.

D  bots:    one seat's setup is played by a given bot (other seat F), then the rest is replayed F vs F
            -> P(that seat wins | its opening), paired against F's own opening on the same board and seat.

usage: python openings.py run OUT.jsonl [boards=300] [m=32] [forced_boards=60] [forced_m=24]
       python openings.py report OUT.jsonl
       python openings.py bots OUT.jsonl boards m SPEC [SPEC ...]   (arena.py specs; F is always included)
       python openings.py botreport OUT.jsonl
"""

import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from catanatron import Game, Color
from catanatron.models.enums import ActionType, SETTLEMENT, CITY
from catanatron.players.value import ValueFunctionPlayer, base_fn, DEFAULT_WEIGHTS

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
VALUE = base_fn(DEFAULT_WEIGHTS)


def setup(seed, force_rank=None):
    g = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=seed)
    first, forced = g.state.colors[0], False
    while g.state.is_initial_build_phase:
        acts, c = g.playable_actions, g.state.current_color()
        if (force_rank is not None and not forced and c == first
                and acts[0].action_type == ActionType.BUILD_SETTLEMENT):
            scores = []
            for a in acts:
                t = g.copy()
                t.execute(a)
                scores.append(VALUE(t, c))
            a = acts[int(np.argsort(-np.array(scores))[force_rank])]
            forced = True
        else:
            a = g.state.current_player().decide(g, acts)
        g.execute(a)
    return g


def features(g, color):
    m = g.state.board.map
    nodes = g.state.buildings_by_color[color][SETTLEMENT] + g.state.buildings_by_color[color][CITY]
    pips = dict.fromkeys(RES, 0.0)
    numbers = set()
    for n in nodes:
        for r, p in m.node_production[n].items():
            pips[r] += 36 * p
        numbers.update(t.number for t in m.adjacent_tiles[n] if t.number)
    port = any(n in ns for ns in m.port_nodes.values() for n in nodes)
    return dict(pips={r: round(v, 2) for r, v in pips.items()}, total=round(sum(pips.values()), 2),
                variety=sum(v > 0 for v in pips.values()), numbers=len(numbers), port=port)


def replay(g, seed, m, color):
    """wins for color over m F-vs-F continuations of g with fresh dice and dev-deck order"""
    wins = 0
    for j in range(m):
        c = g.copy()
        rng = random.Random(seed * 1000 + j)
        c.state.random = c.random = rng
        deck = list(c.state.development_listdeck)
        rng.shuffle(deck)
        c.state.development_listdeck = deck
        while c.winning_color() is None and c.state.num_turns < 1000:
            c.play_tick()
        wins += c.winning_color() == color
    return wins


def job(args):
    seed, force_rank, m = args
    g = setup(seed, force_rank)
    first, second = g.state.colors
    wins = replay(g, seed, m, first)
    return dict(seed=seed, force_rank=force_rank, m=m, wins_first=wins,
                first=features(g, first), second=features(g, second))


_bots = {}


def bot_job(args):
    """setup with `spec` in seat `seat` (0 = first), F in the other; then F-vs-F replays"""
    seed, seat, spec, m = args
    import time
    import torch
    from arena import make
    torch.set_num_threads(1)
    g = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=seed)
    me = g.state.colors[seat]
    if (spec, me) not in _bots:
        _bots[(spec, me)] = make(spec, me)
    bot, ms = _bots[(spec, me)], []
    while g.state.is_initial_build_phase:
        acts = g.playable_actions
        if g.state.current_color() == me:
            t0 = time.perf_counter()
            a = bot.decide(g, acts)
            ms.append(1000 * (time.perf_counter() - t0))
        else:
            a = g.state.current_player().decide(g, acts)
        g.execute(a)
    opp = g.state.colors[1 - seat]
    return dict(seed=seed, seat=seat, spec=spec, m=m, wins=replay(g, seed, m, me), max_ms=max(ms),
                me=features(g, me), opp=features(g, opp))


def bot_report(path):
    rows = [json.loads(l) for l in open(path)]
    base = {(r["seed"], r["seat"]): r["wins"] / r["m"] for r in rows if r["spec"] == "F"}
    specs = sorted({r["spec"] for r in rows}, key=lambda x: x != "F")
    print(f"opening quality: bot plays one seat's setup vs F, rest replayed F vs F "
          f"({len(base) // 2} boards x 2 seats, {rows[0]['m']} replays each)")
    print(f"{'bot':14s} {'win%':>6s} {'vs F opening':>16s} {'better/worse/same':>18s} {'pips':>5s} {'var':>4s} {'sheep':>5s} {'ore':>5s} {'max ms':>7s}")
    for sp in specs:
        R = [r for r in rows if r["spec"] == sp and (r["seed"], r["seat"]) in base]
        p = np.array([r["wins"] / r["m"] for r in R])
        d = p - np.array([base[(r["seed"], r["seat"])] for r in R])
        f = lambda k: np.mean([r["me"][k] for r in R])
        g = lambda k: np.mean([r["me"]["pips"][k] for r in R])
        same = np.mean([r["me"] == next(q for q in rows if q["spec"] == "F" and q["seed"] == r["seed"]
                                          and q["seat"] == r["seat"])["me"] for r in R])
        print(f"{sp:14s} {p.mean():6.1%} {d.mean():+7.1%} ± {1.96 * d.std() / len(d) ** .5:4.1%}"
              f"   {np.mean(d > 0.08):5.0%}/{np.mean(d < -0.08):4.0%}/{same:4.0%} "
              f"{f('total'):5.1f} {f('variety'):4.1f} {g('SHEEP'):5.1f} {g('ORE'):5.1f} {max(r['max_ms'] for r in R):7.0f}")
        for seat in (0, 1):
            ds = [x for x, r in zip(d, R) if r["seat"] == seat]
            print(f"{'':14s}   seat {seat + 1}: {np.mean([x for x, r in zip(p, R) if r['seat'] == seat]):.1%} ({np.mean(ds):+.1%} vs F)")


def logistic(X, y, w8, l2=1e-2, iters=4000, lr=0.2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ ((p - y) * w8) / w8.sum() + l2 * np.r_[w[:-1], 0])
    return w


def report(path):
    rows = [json.loads(l) for l in open(path)]
    B = [r for r in rows if r["force_rank"] is None]
    C = [r for r in rows if r["force_rank"] is not None]
    p = np.array([r["wins_first"] / r["m"] for r in B])
    m = np.mean([r["m"] for r in B])
    pbar = p.mean()
    noise = np.mean(p * (1 - p)) / (m - 1)            # binomial noise in each board's estimate
    true_var = max(p.var() - noise, 0)
    print(f"== B: {len(B)} boards x {m:.0f} replays (F vs F)")
    print(f"seat: first player wins {pbar:.1%} (± {1.96 * p.std() / len(p) ** .5:.1%})")
    print(f"spread of P(first wins | opening): sd {p.std():.3f} raw, {true_var ** .5:.3f} after removing replay noise")
    print(f"share of the game's win/loss uncertainty already decided by the opening: {true_var / (pbar * (1 - pbar)):.0%}")
    print(f"boards where the opening gives one side >= 65%: {np.mean((p >= .65) | (p <= .35)):.0%};  >= 75%: "
          f"{np.mean((p >= .75) | (p <= .25)):.0%}")
    names = ["total pips", "resource variety", "distinct numbers", "port"] + [f"{r.lower()} pips" for r in RES]
    d = lambda r, f: (r["first"][f] - r["second"][f]) if f != "port" else float(r["first"]["port"]) - float(r["second"]["port"])
    X = np.array([[d(r, "total"), d(r, "variety"), d(r, "numbers"), d(r, "port")]
                  + [r["first"]["pips"][x] - r["second"]["pips"][x] for x in RES] for r in B])
    corr = [np.corrcoef(X[:, i], p)[0, 1] for i in range(X.shape[1])]
    print("correlation of (first - second) opening feature with P(first wins):")
    for n, c in sorted(zip(names, corr), key=lambda z: -abs(z[1])):
        print(f"  {n:18s} {c:+.2f}")
    # one regression on all replays (each board's replays weighted equally)
    y = np.repeat(p, 1)
    w = logistic(X, y, np.full(len(y), 1.0))
    print("logistic fit, standardised coefficients (effect of +1 sd difference, holding the others fixed):")
    for n, c in sorted(zip(names, w[:-1]), key=lambda z: -abs(z[1])):
        print(f"  {n:18s} {c:+.2f}")
    if C:
        by = {}
        for r in C:
            by.setdefault(r["seed"], {})[r["force_rank"]] = r["wins_first"] / r["m"]
        ranks = sorted({r["force_rank"] for r in C})
        M = np.array([[by[s][k] for k in ranks] for s in by if len(by[s]) == len(ranks)])
        mc = np.mean([r["m"] for r in C])
        print(f"\n== C: {len(M)} boards, first settlement forced to F's rank 1..{len(ranks)} corner, {mc:.0f} replays each")
        for k, v in zip(ranks, M.mean(0)):
            print(f"  rank {k + 1} corner: first player wins {v:.1%}")
        rng_ = M.max(1) - M.min(1)
        print(f"  best minus worst of the top {len(ranks)} corners, per board: mean {rng_.mean():.1%} "
              f"(replay noise alone would give ~{np.sqrt(2 * .25 / mc) * 1.2:.0%})")
        print(f"  F's own pick (rank 1) is the best of the {len(ranks)}: {np.mean(M.argmax(1) == 0):.0%} of boards")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2])
        sys.exit()
    if sys.argv[1] == "botreport":
        bot_report(sys.argv[2])
        sys.exit()
    if sys.argv[1] == "bots":
        out, nb, m = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        specs = ["F"] + [x for x in sys.argv[5:] if x != "F"]
        jobs = [(70000 + i, seat, sp, m) for i in range(nb) for seat in (0, 1) for sp in specs]
        random.Random(0).shuffle(jobs)
        W = int(os.environ.get("WORKERS", 14))
        with Pool(W) as pool, open(out, "w") as f:
            for done, r in enumerate(pool.imap_unordered(bot_job, jobs), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if done % 50 == 0:
                    print(f"{done}/{len(jobs)} jobs", flush=True)
        bot_report(out)
        sys.exit()
    out = sys.argv[2]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    nf = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    mf = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    jobs = [(50000 + i, None, m) for i in range(nb)] + [(60000 + i, r, mf) for i in range(nf) for r in range(5)]
    W = int(os.environ.get("WORKERS", 14))
    done = 0
    with Pool(W) as pool, open(out, "w") as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} jobs", flush=True)
    report(out)
