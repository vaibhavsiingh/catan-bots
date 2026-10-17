"""Generate behavior-cloning data: a teacher bot plays, we record (state, legal actions, teacher choice, outcome).

usage: python gen_data.py OUT.npz NUM_GAMES [teacher=F|AB] [eps=0.1] [players=2]
"""

import os
import sys
import random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from encode import encode_state, encode_actions
from catanatron import Game, Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def make_recorder(base_cls):
    class Recorder(base_cls):
        def __init__(self, color, eps, buf):
            super().__init__(color)
            self.eps, self.buf = eps, buf

        def decide(self, game, playable_actions):
            best = super().decide(game, playable_actions)
            if len(playable_actions) > 1:
                acts = list(playable_actions)
                self.buf.append((self.color, encode_state(game, self.color),
                                 encode_actions(game, self.color, acts), acts.index(best)))
            if self.eps and random.random() < self.eps:
                return random.choice(playable_actions)
            return best

    return Recorder


def play_games(args):
    n, seed, teacher, eps, nplayers = args
    random.seed(seed)
    cls = make_recorder(ValueFunctionPlayer if teacher == "F" else AlphaBetaPlayer)
    out = []
    for i in range(n):
        buf = []
        players = [cls(c, eps, buf) for c in COLORS[:nplayers]]
        game = Game(players, seed=seed * 100000 + i)
        game.play()
        w = game.winning_color()
        for color, s, a, y in buf:
            out.append((s, a, y, 0.5 if w is None else float(w == color)))
    return out


def pack(samples, max_actions=None):
    A = max_actions or max(len(s[1][0]) for s in samples)
    N = len(samples)
    tile = np.stack([s[0][0] for s in samples]).astype(np.float16)
    node = np.stack([s[0][1] for s in samples]).astype(np.float16)
    edge = np.stack([s[0][2] for s in samples]).astype(np.float16)
    player = np.stack([s[0][3] for s in samples]).astype(np.float16)
    glob = np.stack([s[0][4] for s in samples]).astype(np.float16)
    a_int = np.full((N, A, 3), -1, np.int16)
    a_res = np.zeros((N, A, 10), np.float16)
    n_act = np.zeros(N, np.int16)
    for i, s in enumerate(samples):
        k = len(s[1][0])
        a_int[i, :k], a_res[i, :k], n_act[i] = s[1][0], s[1][1], k
    y = np.array([s[2] for s in samples], np.int16)
    v = np.array([s[3] for s in samples], np.float16)
    return dict(tile=tile, node=node, edge=edge, player=player, glob=glob,
                a_int=a_int, a_res=a_res, n_act=n_act, y=y, v=v)


if __name__ == "__main__":
    out, num = sys.argv[1], int(sys.argv[2])
    teacher = sys.argv[3] if len(sys.argv) > 3 else "F"
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    nplayers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    workers = int(os.environ.get("WORKERS", 15))
    per = [num // workers + (1 if i < num % workers else 0) for i in range(workers)]
    base = random.randrange(10**6)
    with Pool(workers) as pool:
        res = pool.map(play_games, [(per[i], base + i, teacher, eps, nplayers) for i in range(workers)])
    samples = [s for r in res for s in r]
    print(f"{len(samples)} samples from {num} games; max actions {max(len(s[1][0]) for s in samples)}")
    np.savez_compressed(out, **pack(samples))
