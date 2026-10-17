"""Value-model data: states sampled at every tick, from BOTH players' perspectives.

Labels per (state, perspective):
  win     1 / 0 / 0.5 (no winner)
  dvp     change in (my VP - opp VP) over the next H turns (short-horizon, action-sensitive)
  margin  final (my VP - opp VP) / 10

usage: python gen_value.py OUT.npz NUM_GAMES [sample_p=0.35] [eps=0.15]
"""

import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catanatron import Game, Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points
from encode import encode_state

H = 4  # horizon in turns for dvp


class EpsF(ValueFunctionPlayer):
    def __init__(self, color, eps):
        super().__init__(color)
        self.eps = eps

    def decide(self, game, playable_actions):
        if random.random() < self.eps:
            return random.choice(playable_actions)
        return super().decide(game, playable_actions)


def play_games(args):
    n, seed, p, eps = args
    random.seed(seed)
    out = []
    for i in range(n):
        colors = [Color.RED, Color.BLUE]
        g = Game([EpsF(c, eps) for c in colors], seed=seed * 100000 + i)
        recs, vp_by_turn = [], {}
        while g.winning_color() is None and g.state.num_turns < 1000:
            t = g.state.num_turns
            vp_by_turn[t] = {c: get_actual_victory_points(g.state, c) for c in colors}
            if random.random() < p:
                for c in colors:
                    recs.append((encode_state(g, c), c, t))
            g.play_tick()
        w = g.winning_color()
        end = {c: get_actual_victory_points(g.state, c) for c in colors}
        last_t = g.state.num_turns
        vp_by_turn[last_t] = end
        for s, c, t in recs:
            o = colors[1] if c == colors[0] else colors[0]
            t2 = min(t + H, last_t)
            while t2 not in vp_by_turn:
                t2 -= 1
            now = vp_by_turn[t][c] - vp_by_turn[t][o]
            later = vp_by_turn[t2][c] - vp_by_turn[t2][o]
            win = 0.5 if w is None else float(w == c)
            out.append((s, win, later - now, (end[c] - end[o]) / 10))
    return out


if __name__ == "__main__":
    path, num = sys.argv[1], int(sys.argv[2])
    p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    W = int(os.environ.get("WORKERS", 15))
    per = [num // W + (1 if i < num % W else 0) for i in range(W)]
    base = random.randrange(10**6)
    with Pool(W) as pool:
        res = pool.map(play_games, [(per[i], base + i, p, eps) for i in range(W)])
    S = [x for r in res for x in r]
    st = lambda i: np.stack([x[0][i] for x in S]).astype(np.float16)
    np.savez_compressed(path, tile=st(0), node=st(1), edge=st(2), player=st(3), glob=st(4),
                        win=np.array([x[1] for x in S], np.float16),
                        dvp=np.array([x[2] for x in S], np.float16),
                        margin=np.array([x[3] for x in S], np.float16))
    print(f"{len(S)} samples from {num} games")
