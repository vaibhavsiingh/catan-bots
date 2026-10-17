"""Where does the TSRCH bot choose differently from our PPO policy, F and plain V2-Tuned?

Plays TSRCH vs F; at every TSRCH decision with a real choice, also asks PPO v2, F and V2-Tuned greedy
(no net) what they would play in the same position, and logs the category of the decision, whether
TSRCH's look-ahead ran, and TSRCH's time.

usage: python tsrch_probe.py run OUT.jsonl [games=100]
       python tsrch_probe.py report OUT.jsonl
"""

import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_EXT = os.environ.get("EXTERNAL_BOTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../external"))
sys.path.insert(0, _EXT)
import numpy as np
import torch
from catanatron import Game, Color
from catanatron.models.enums import ActionType, ActionPrompt
from catanatron.players.value import ValueFunctionPlayer

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")


def category(game, action):
    pr = game.state.current_prompt
    if pr == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return "setup settlement"
    if pr == ActionPrompt.BUILD_INITIAL_ROAD:
        return "setup road"
    if pr == ActionPrompt.MOVE_ROBBER:
        return "robber"
    if pr == ActionPrompt.DISCARD:
        return "discard"
    return "turn: " + action.action_type.value.lower()


class Probe:
    """wraps TSRCH; on each real decision records what the other bots would do"""

    def __init__(self, color):
        from gtblend.player import TSRCHPlayer, V2T
        from gtblend import adv
        from tf_player import TransformerPlayer
        self.color = color
        self.t = TSRCHPlayer(color)
        self.others = {"ppo": TransformerPlayer(color, ckpt=_D + "ppo2_it20.pt"),
                       "F": ValueFunctionPlayer(color), "v2t": adv.V2Greedy(color, V2T)}
        self.rows = []
        self.sims = 0
        orig = self.t._sim

        def counted(*a, **k):
            self.sims += 1
            return orig(*a, **k)
        self.t._sim = counted

    def decide(self, game, acts):
        if len(acts) == 1:
            return acts[0]
        s0, t0 = self.sims, time.perf_counter()
        a = self.t.decide(game, acts)
        ms = 1000 * (time.perf_counter() - t0)
        row = dict(cat=category(game, a), n=len(acts), ms=ms, look=self.sims > s0, turn=game.state.num_turns)
        for k, p in self.others.items():
            b = p.decide(game.copy(), acts)
            row[k] = b == a
            row[k + "_cat"] = category(game, b)
        self.rows.append(row)
        return a


def job(seed):
    torch.set_num_threads(1)
    from catanatron.models.player import Player

    class P(Player):
        def __init__(self, color):
            super().__init__(color)
            self.probe = Probe(color)

        def decide(self, game, acts):
            return self.probe.decide(game, acts)

    me = P(Color.RED)
    players = [me, ValueFunctionPlayer(Color.BLUE)]
    if seed % 2:
        players.reverse()
    g = Game(players, seed=seed)
    g.play()
    won = g.winning_color() == Color.RED
    for r in me.probe.rows:
        r["won"] = won
        r["seed"] = seed
    return me.probe.rows


def report(path):
    rows = [json.loads(l) for l in open(path)]
    games = {r["seed"]: r["won"] for r in rows}
    print(f"{len(games)} games TSRCH vs F, TSRCH won {np.mean(list(games.values())):.0%}; {len(rows)} real decisions")
    print(f"look-ahead ran on {np.mean([r['look'] for r in rows]):.0%} of decisions; "
          f"ms/decision mean {np.mean([r['ms'] for r in rows]):.0f}, with look-ahead "
          f"{np.mean([r['ms'] for r in rows if r['look']]):.0f}, without {np.mean([r['ms'] for r in rows if not r['look']]):.0f}")
    print(f"real decisions per game {len(rows) / len(games):.0f}\n")
    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    print(f"{'TSRCH chose':28s} {'count':>6s} {'look':>5s}   agrees with:  {'ppo':>5s} {'F':>5s} {'v2t':>5s}")
    for c, rs in sorted(by.items(), key=lambda z: -len(z[1])):
        print(f"{c:28s} {len(rs):6d} {np.mean([r['look'] for r in rs]):5.0%}   {'':13s}"
              + " ".join(f"{np.mean([r[k] for r in rs]):5.0%}" for k in ("ppo", "F", "v2t")))
    print(f"{'ALL':28s} {len(rows):6d} {'':5s}   {'':13s}" + " ".join(f"{np.mean([r[k] for r in rows]):5.0%}" for k in ("ppo", "F", "v2t")))
    for k in ("ppo", "F"):
        conf = defaultdict(int)
        for r in rows:
            if not r[k]:
                conf[(r["cat"], r[k + "_cat"])] += 1
        print(f"\nmost common disagreements, TSRCH -> {k}:")
        for (a, b), n in sorted(conf.items(), key=lambda z: -z[1])[:12]:
            print(f"  {n:5d}  TSRCH {a:26s} {k}: {b}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2])
        sys.exit()
    out, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100
    with Pool(int(os.environ.get("WORKERS", 14))) as pool, open(out, "w") as f:
        for i, rows in enumerate(pool.imap_unordered(job, range(31001, 31001 + n)), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{n} games", flush=True)
    report(out)
