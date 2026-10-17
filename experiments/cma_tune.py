"""CMA-ES on the heuristics bot's weights (heur.py), objective = win rate against chosen opponents.

Each generation: every candidate plays the same G boards from both seats (new boards each generation).
Weights are searched as w = w0 + x * scale, scale = max(|w0|, 0.2) * 0.5, starting at x = 0.
Every EVAL_EVERY generations the CMA mean is checked on fixed held-out boards.

usage: python cma_tune.py OUT_DIR [gens=12] [boards=100] [opps=F]   (opps comma-separated arena specs)
"""

import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cma
from catanatron import Game, Color
from heur import HeurPlayer, WEIGHTS

KEYS = list(WEIGHTS)
W0 = np.array([WEIGHTS[k] for k in KEYS])
SCALE = np.maximum(np.abs(W0), 0.2) * 0.5
EVAL_EVERY, EVAL_BOARDS = 3, 200


def to_w(x):
    return dict(zip(KEYS, (W0 + np.asarray(x) * SCALE).tolist()))


_opp = {}


def play(args):
    """one board, our bot in seat `seat`; returns 1 if we won"""
    w, opp_spec, seed, seat = args
    import torch
    torch.set_num_threads(1)
    from arena import make
    random.seed(seed * 2 + seat)
    me = HeurPlayer(Color.RED, w)
    if opp_spec not in _opp:
        _opp[opp_spec] = make(opp_spec, Color.BLUE)
    players = [me, _opp[opp_spec]]
    if seat:
        players.reverse()
    g = Game(players, seed=seed)
    g.play()
    return float(g.winning_color() == Color.RED)


def score(pool, ws, opps, seeds):
    jobs = [(w, o, s, seat) for w in ws for o in opps for s in seeds for seat in (0, 1)]
    r = np.array(pool.map(play, jobs, chunksize=4)).reshape(len(ws), -1)
    return r.mean(1)


if __name__ == "__main__":
    out = sys.argv[1]
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    opps = sys.argv[4].split(",") if len(sys.argv) > 4 else ["F"]
    os.makedirs(out, exist_ok=True)
    es = cma.CMAEvolutionStrategy(np.zeros(len(KEYS)), 0.5, {"seed": 1, "verbose": -9})
    held = list(range(900000, 900000 + EVAL_BOARDS))
    log = open(os.path.join(out, "log.txt"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    with Pool(int(os.environ.get("WORKERS", 14))) as pool:
        base = score(pool, [dict(WEIGHTS)], opps, held)[0]
        say(f"start weights on held-out boards: {base:.3f} vs {opps} ({2 * EVAL_BOARDS * len(opps)} games)")
        for gen in range(1, gens + 1):
            t0 = time.time()
            xs = es.ask()
            seeds = list(range(100000 * gen, 100000 * gen + nb))
            wr = score(pool, [to_w(x) for x in xs], opps, seeds)
            es.tell(xs, (-wr).tolist())
            say(f"gen {gen:3d}: candidates mean {wr.mean():.3f} best {wr.max():.3f}  sigma {es.sigma:.3f}  "
                f"{time.time() - t0:.0f}s")
            json.dump(to_w(es.mean), open(os.path.join(out, "mean.json"), "w"), indent=1)
            if gen % EVAL_EVERY == 0 or gen == gens:
                h = score(pool, [to_w(es.mean)], opps, held)[0]
                say(f"   held-out: CMA mean weights {h:.3f}  (start {base:.3f})")
                json.dump(to_w(es.mean), open(os.path.join(out, f"mean_g{gen}.json"), "w"), indent=1)
