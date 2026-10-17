"""Which move is better where PPO v2 and TSRCH disagree?

Plays PPO v2 vs F. At each PPO decision TSRCH is asked too; on a disagreement the game is forked: one
copy plays PPO's move, the other TSRCH's, and each is played to the end N times with PPO continuing
for us and F for the opponent, same dice / dev-deck seed for both forks (common random numbers).
Then PPO's move is played and the game goes on.

usage: python judge.py run OUT.jsonl [games=150] [N=16]
       python judge.py report OUT.jsonl
"""

import json
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("EXTERNAL_BOTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../external")))
import numpy as np
import torch
from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from tsrch_probe import category

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")


def playout(game, action, me, seed, ppo, f):
    g = game.copy()
    rng = random.Random(seed)
    g.state.random = g.random = rng
    deck = list(g.state.development_listdeck)
    rng.shuffle(deck)
    g.state.development_listdeck = deck
    g.state.players = [ppo if c == me else f for c in g.state.colors]
    g.execute(action)
    while g.winning_color() is None and g.state.num_turns < 1000:
        g.play_tick()
    return g.winning_color() == me


class Judge(Player):
    def __init__(self, color, n, rows):
        super().__init__(color)
        from gtblend.player import TSRCHPlayer
        from tf_player import TransformerPlayer
        self.ppo = TransformerPlayer(color, ckpt=_D + "ppo2_it20.pt")
        self.tsrch = TSRCHPlayer(color)
        self.n, self.rows = n, rows
        self.opp = None

    def decide(self, game, acts):
        if len(acts) == 1:
            return acts[0]
        a = self.ppo.decide(game, acts)
        b = self.tsrch.decide(game.copy(), acts)
        if a != b:
            if self.opp is None:
                self.opp = ValueFunctionPlayer(next(c for c in game.state.colors if c != self.color))
            seeds = [random.randrange(1 << 30) for _ in range(self.n)]
            wa = sum(playout(game, a, self.color, s, self.ppo, self.opp) for s in seeds)
            wb = sum(playout(game, b, self.color, s, self.ppo, self.opp) for s in seeds)
            ps = game.state.player_state
            i = game.state.color_to_index[self.color]
            self.rows.append(dict(cat_ppo=category(game, a), cat_tsrch=category(game, b), n=self.n,
                                  w_ppo=wa, w_tsrch=wb, turn=game.state.num_turns,
                                  vp=ps[f"P{i}_VICTORY_POINTS"], opp_vp=ps[f"P{1 - i}_VICTORY_POINTS"],
                                  n_acts=len(acts)))
        return a


def job(args):
    seed, n = args
    torch.set_num_threads(1)
    random.seed(seed)
    rows = []
    me = Judge(Color.RED, n, rows)
    players = [me, ValueFunctionPlayer(Color.BLUE)]
    if seed % 2:
        players.reverse()
    g = Game(players, seed=seed)
    g.play()
    for r in rows:
        r["seed"] = seed
        r["won"] = g.winning_color() == Color.RED
    return rows


def report(path):
    rows = [json.loads(l) for l in open(path)]
    n = rows[0]["n"]
    print(f"{len({r['seed'] for r in rows})} games PPO v2 vs F, {len(rows)} disagreements with TSRCH, "
          f"{n} playouts per move\n")

    def show(title, key):
        by = defaultdict(list)
        for r in rows:
            by[key(r)].append((r["w_tsrch"] - r["w_ppo"]) / r["n"])
        print(f"{title:44s} {'count':>6s} {'TSRCH minus PPO':>17s}  {'TSRCH better / worse':>20s}")
        for k, d in sorted(by.items(), key=lambda z: -len(z[1])):
            if len(d) < 15:
                continue
            d = np.array(d)
            print(f"{k:44s} {len(d):6d} {d.mean():+8.1%} ± {1.96 * d.std() / len(d) ** .5:4.1%}"
                  f"    {np.mean(d >= 0.25):5.0%} / {np.mean(d <= -0.25):4.0%}")
        d = np.array([(r["w_tsrch"] - r["w_ppo"]) / r["n"] for r in rows])
        print(f"{'ALL':44s} {len(d):6d} {d.mean():+8.1%} ± {1.96 * d.std() / len(d) ** .5:4.1%}\n")

    show("by TSRCH's move", lambda r: r["cat_tsrch"])
    show("by (TSRCH's move -> PPO's move)", lambda r: f"{r['cat_tsrch'][:20]} -> {r['cat_ppo'][:20]}")
    show("by game phase", lambda r: "setup" if r["cat_tsrch"].startswith("setup")
         else f"max VP {min(max(r['vp'], r['opp_vp']) // 3 * 3, 9)}-{min(max(r['vp'], r['opp_vp']) // 3 * 3 + 2, 11)}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2])
        sys.exit()
    out = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    with Pool(int(os.environ.get("WORKERS", 14))) as pool, open(out, "w") as f:
        for i, rows in enumerate(pool.imap_unordered(job, [(52001 + k, n) for k in range(games)]), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{i}/{games} games, {sum(1 for _ in open(out))} disagreements", flush=True)
    report(out)
