"""Cross-entropy-method tuning of value-function weights (in log-space) against the stock ValueFunctionPlayer.

usage: python tune.py [iters=6] [pop=12] [games=150]
"""

import json
import math
import random
import sys
from multiprocessing import Pool

import numpy as np
from catanatron import Game, Color
from catanatron.players.value import ValueFunctionPlayer, DEFAULT_WEIGHTS

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from my_value import MyValuePlayer, WEIGHTS_FILE

KEYS = [k for k, v in DEFAULT_WEIGHTS.items() if v != 0 and k != "public_vps"]


def to_weights(z):
    w = dict(DEFAULT_WEIGHTS)
    for k, x in zip(KEYS, z):
        w[k] = math.copysign(math.exp(x), DEFAULT_WEIGHTS[k])
    return w


def play(args):
    z, seeds = args
    w = to_weights(z)
    wins = 0
    for s in seeds:
        me, opp = MyValuePlayer(Color.RED, w), ValueFunctionPlayer(Color.BLUE)
        g = Game([me, opp] if s % 2 == 0 else [opp, me], seed=s)
        g.play()
        wins += g.winning_color() == Color.RED
    return wins


def score(pool, z, seeds, W=15):
    chunks = [seeds[i::W] for i in range(W)]
    return sum(pool.map(play, [(z, c) for c in chunks])) / len(seeds)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    mu = np.array([math.log(abs(DEFAULT_WEIGHTS[k])) for k in KEYS])
    sigma = np.full(len(KEYS), 1.0)
    rng = np.random.default_rng(0)
    best = (0.5, mu.copy())
    with Pool(15) as pool:
        for it in range(iters):
            seeds = list(range(it * 10000, it * 10000 + games))  # common seeds within an iteration
            zs = [mu] + [mu + sigma * rng.standard_normal(len(KEYS)) for _ in range(pop - 1)]
            scores = [score(pool, z, seeds) for z in zs]
            order = np.argsort(scores)[::-1]
            elite = np.array([zs[i] for i in order[: max(2, pop // 4)]])
            mu, sigma = elite.mean(0), np.maximum(elite.std(0), 0.15)
            print(f"iter {it}: mean-candidate {scores[0]:.3f} best {scores[order[0]]:.3f} "
                  f"top4 {[round(scores[i],3) for i in order[:4]]}", flush=True)
        # confirm the final mean on fresh seeds
        final = score(pool, mu, list(range(900000, 900000 + 600)))
        print(f"final mean weights vs F on 600 fresh games: {final:.3f}")
    w = to_weights(mu)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump({k: w[k] for k in KEYS}, f, indent=1)
    print(json.dumps({k: f"{w[k]:.3g}" for k in KEYS}))
