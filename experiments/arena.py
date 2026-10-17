"""Parallel head-to-head evaluation with seat swapping.

usage: python arena.py A B [games=200]
  bots: R, W, F, AB, TF:path/to/ckpt.pt, MYV
"""

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from catanatron import Game, Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.player import Player


_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")
_P = f"{_D}ppo2_it20.pt:{_D}val2.pt"
ALIASES = {  # named bots (search settings: k candidates : n simulations : conf : policy : value : ...)
    "OLD": f"CTR:4:4:0.9:{_P}:dev",
    "OLD450": f"CTR:4:4:0.9:{_P}:dev:budget=450",  # OLD with a 450 ms budget (leaf eval included)
    "NEW450": f"CTR:4:4:0.9:{_P}:none:road=3:depth2=1:budget=450",
    "OLD450_CTRS": f"CTRS:4:4:0.9:{_P}:dev:budget=450",  # opponent modelled as our own policy
    "NEW450_CTRS": f"CTRS:4:4:0.9:{_P}:none:road=3:depth2=1:budget=450",
    "BIG": f"CTR:6:8:0.98:{_P}:none:road=3:depth2=1:d2k=3:budget=700",
    "MEGATRON": f"CTR:6:8:0.9:{_P}:none:road=3:depth2=1:d2k=4:budget=700",
    "MEGASTART": "SETUP:NETSEARCH|NEW450",  # NETSEARCH openings (best in openings.py bots), NEW450 afterwards
}


class SetupSplit(Player):
    """one bot for the initial placements, another for the rest of the game"""

    def __init__(self, setup, rest):
        super().__init__(rest.color)
        self.setup, self.rest = setup, rest

    def decide(self, game, playable_actions):
        bot = self.setup if game.state.is_initial_build_phase else self.rest
        return bot.decide(game, playable_actions)

    def before(self, game):
        self.setup.before(game)
        self.rest.before(game)

    def step(self, game_before_action, action):
        self.setup.step(game_before_action, action)
        self.rest.step(game_before_action, action)

    def after(self, game):
        self.setup.after(game)
        self.rest.after(game)


def make(spec, color):
    spec = ALIASES.get(spec, spec)
    if spec.startswith("SETUP:"):  # SETUP:setup_spec|rest_spec
        a, b = spec[6:].split("|")
        return SetupSplit(make(a, color), make(b, color))
    if spec == "H" or spec.startswith("H:"):  # H[:weights.json]  plain heuristics bot (heur.py)
        from heur import HeurPlayer, load
        return HeurPlayer(color, load(spec[2:] or None))
    if spec == "R":
        return RandomPlayer(color)
    if spec == "W":
        return WeightedRandomPlayer(color)
    if spec == "F":
        return ValueFunctionPlayer(color)
    if spec == "AB":
        return AlphaBetaPlayer(color)
    if spec.startswith("OWS:"):
        from ows_player import from_spec
        return from_spec(spec, color)
    if spec.startswith(("CTR:", "CTRS:")):  # CTR[S]:k:n:conf:policy:value[:none|dev|all]  opponent modelled as F (CTR) or as our policy (CTRS)
        from counter_player import CounterPlayer
        parts = spec.split(":")
        extra = {}  # trailing key=value options: road=3, depth2=1, budget=450
        while "=" in parts[-1]:
            key, val = parts.pop().split("=")
            extra[{"budget": "budget_ms"}.get(key, key)] = bool(int(val)) if key == "depth2" else float(val)
        # e.g. CTR:6:8:0.98:policy:value:none:road=3:depth2=1:d2k=3:budget=800
        hidden = parts.pop() if parts[-1] in ("none", "dev", "all") else "none"
        kind, k, n, conf, path, vpath = parts
        return CounterPlayer(color, path, vpath, k=int(k), n=int(n), conf=float(conf),
                             opp="SELF" if kind == "CTRS" else "F", hidden=hidden, **extra)
    if spec.startswith("LF:"):  # LF:weights.json  learned scoring function, 1-ply greedy like F
        from learned_value import LearnedValuePlayer
        return LearnedValuePlayer(color, spec[3:])
    if spec.startswith("GTBLEND:"):  # GTBLEND:TSRCH | BTUNE | V2T | V2  graph transformer + V2-Tuned blend (package gtblend, not included)
        repo = os.environ.get("EXTERNAL_BOTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../external"))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from gtblend.player import PLAYERS, V2T
        from gtblend import adv
        if spec[8:] in ("V2T", "V2"):  # V2-Tuned / default V2 linear evaluation alone, 1-ply expectimax, no net
            return adv.V2Greedy(color, V2T if spec[8:] == "V2T" else None)
        return PLAYERS[spec[8:]](color)
    if spec.startswith("NETSEARCH"):  # NETSEARCH[:variant]  transformer net + opponent-aware search (package bot, not included)
        repo = os.environ.get("EXTERNAL_BOTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../external"))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from bot.config import Config
        from bot.search import SearchPlayer as NetSearch
        return NetSearch(color, cfg=Config.load(spec[10:]))
    if spec == "MYV":
        from my_value import MyValuePlayer
        return MyValuePlayer(color)
    if spec.startswith("TFV:"):  # TFV:k:beta:lam:policy_ckpt:value_ckpt  (dedicated value net)
        from search_player import SearchPlayer
        _, k, beta, lam, path, vpath = spec.split(":", 5)
        return SearchPlayer(color, path, k=int(k), beta=float(beta), vnet=vpath, lam=float(lam))
    if spec.startswith("TFS:"):  # TFS:k:beta:ckpt  (value-head lookahead)
        from search_player import SearchPlayer
        _, k, beta, path = spec.split(":", 3)
        return SearchPlayer(color, path, k=int(k), beta=float(beta))
    if spec.startswith("TFT:"):  # TFT:temperature:ckpt
        from tf_player import TransformerPlayer
        _, temp, path = spec.split(":", 2)
        return TransformerPlayer(color, ckpt=path, temperature=float(temp))
    if spec.startswith("TF:"):
        from tf_player import TransformerPlayer
        return TransformerPlayer(color, ckpt=spec[3:])
    raise ValueError(spec)


def run(args):
    a, b, seeds = args
    torch.set_num_threads(1)
    res = {"A": 0, "B": 0, "draw": 0, "tA": 0.0, "tB": 0.0, "nA": 0, "nB": 0, "mA": 0.0, "mB": 0.0, "sA": 0, "sB": 0}
    for s in seeds:
        swap = s % 2 == 1  # alternate who sits first
        ca, cb = (Color.BLUE, Color.RED) if swap else (Color.RED, Color.BLUE)
        pa, pb = make(a, ca), make(b, cb)
        for p, k in ((pa, "A"), (pb, "B")):  # time each bot's decisions
            orig = p.decide

            def timed(game, acts, orig=orig, k=k):
                t = time.perf_counter()
                r = orig(game, acts)
                dt = time.perf_counter() - t
                res["t" + k] += dt
                res["n" + k] += 1
                res["m" + k] = max(res["m" + k], dt)
                res["s" + k] += dt > 1.0  # over the competition's 1000 ms limit
                return r
            p.decide = timed
        players = [pb, pa] if swap else [pa, pb]
        g = Game(players, seed=s)
        g.play()
        w = g.winning_color()
        res["A" if w == ca else "B" if w == cb else "draw"] += 1
    return res


def summary(a, b, tot):
    wr = tot["A"] / max(1, tot["A"] + tot["B"])
    se = (wr * (1 - wr) / max(1, tot["A"] + tot["B"])) ** 0.5
    ng = max(1, tot["A"] + tot["B"] + tot["draw"])
    return (f"{a} vs {b}: {tot['A']}-{tot['B']} (draws {tot['draw']})  winrate A = {wr:.3f} ± {1.96*se:.3f}  "
            f"ms/decision A={1000*tot['tA']/max(1,tot['nA']):.2f} B={1000*tot['tB']/max(1,tot['nB']):.2f}  "
            f"max ms A={1000*tot['mA']:.0f} B={1000*tot['mB']:.0f}  >1s A={tot['sA']} B={tot['sB']}  "
            f"ms/game A={1000*tot['tA']/ng:.0f} B={1000*tot['tB']/ng:.0f} (harness clock ~1000 ms/game)")


def merge(rs):
    return {k: (max if k in ("mA", "mB") else sum)(r[k] for r in rs) for k in rs[0]}


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    W = int(os.environ.get("WORKERS", 15))
    every = int(os.environ.get("LOG_EVERY", 0))  # print a running total every N finished games
    rs = []
    with Pool(W) as pool:
        seed0 = int(os.environ.get("SEED0", 0))  # board seeds seed0 .. seed0+n-1
        for r in pool.imap_unordered(run, [(a, b, [s]) for s in range(seed0, seed0 + n)]):
            rs.append(r)
            if every and len(rs) % every == 0 and len(rs) < n:
                print(f"[{time.strftime('%H:%M')}] {len(rs)}/{n} games  " + summary(a, b, merge(rs)), flush=True)
    print(summary(a, b, merge(rs)), flush=True)
