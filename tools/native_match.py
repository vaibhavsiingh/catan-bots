"""A harness agent against one of catanatron's bots playing natively, with the TRUE game (it sees the opponent's
development cards, as in catanatron's own matches). The upper-bound test: the harness masks the game for both
seats, which handicaps search bots like AlphaBeta.

Our agent gets exactly what the harness gives it (a masked copy of the game, wrapped in Obs). Boards, seats and
dice follow the harness (game i on board seed0 + i//2, even i: our agent sits first, dice from match.dice_seed).
Both players run in this process, each with its own saved `random` state, so neither moves the dice. No clock.

usage: python tools/native_match.py OUT AGENT.py [--games 300] [--seed0 550000] [--par 8] [--chunk 20]
  the opponent is chosen with tools/cat_agent.py's environment variables (CAT=AB, CAT_DEPTH=2, ...).
  Finished chunks in OUT/ are reused; OUT/arm.json records the settings and a mismatch is refused.
"""

import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import harness  # noqa: E402,F401  the determinism fix
from harness.host import AGENT_RANDOM_SEED, _load_agent  # noqa: E402
from harness.match import COLORS, DEFAULT_KEY, dice_seed  # noqa: E402
from harness.obs import Obs, masked_copy  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Player, RandomPlayer  # noqa: E402

import cat_agent  # noqa: E402


class Own(Player):
    """Runs `inner` (a catanatron Player, or a harness Agent when is_agent=True) on its own `random` state."""

    def __init__(self, color, inner, is_agent):
        super().__init__(color)
        self.inner, self.is_agent = inner, is_agent
        self.mask_rng = random.Random(0)
        self.rs = random.Random(AGENT_RANDOM_SEED).getstate()

    def new_game(self, seed):
        self.rs = random.Random(seed).getstate()
        self.mask_rng = random.Random(seed ^ 0x5EED)
        if not self.is_agent:
            self.inner = cat_agent.make(self.color)

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        outer = random.getstate()
        random.setstate(self.rs)
        try:
            if self.is_agent:
                view = masked_copy(game, self.color, self.mask_rng)
                view.state.players = [RandomPlayer(c) for c in view.state.colors]
                view.seed = 0
                i = self.inner.decide(Obs(view, self.color), actions)
                return actions[i] if type(i) is int and 0 <= i < len(actions) else random.choice(actions)
            return self.inner.decide(game, actions)
        finally:
            self.rs = random.getstate()
            random.setstate(outer)


def play(job):
    agent_path, seed0, lo, hi, out = job
    os.chdir(os.path.dirname(agent_path))
    random.seed(AGENT_RANDOM_SEED)
    ours = Own(COLORS[0], _load_agent(agent_path)(), is_agent=True)
    opp = Own(COLORS[1], None, is_agent=False)
    records = []
    for gi in range(lo, hi):
        board_seed = seed0 + gi // 2
        swap = gi % 2 == 1
        ours.color, opp.color = (COLORS[1], COLORS[0]) if swap else (COLORS[0], COLORS[1])
        s = (board_seed * 1_000_003 + gi) & 0xFFFFFFFF
        ours.new_game(s)
        opp.new_game(s ^ 0xABCDEF)
        random.seed(board_seed)
        game = Game([opp, ours] if swap else [ours, opp], seed=board_seed, vps_to_win=10)
        random.seed(dice_seed(board_seed, gi // 2, DEFAULT_KEY))
        w = game.play()
        records.append(dict(seed0=seed0, game=gi, board_seed=board_seed, turns=game.state.num_turns,
                            winner=None if w is None else ("a" if w == ours.color else "b")))
    with open(out + ".tmp", "w") as f:
        json.dump(records, f)
    os.replace(out + ".tmp", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("agent")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=550000)
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()
    out, agent = os.path.abspath(a.out), os.path.abspath(a.agent)
    os.makedirs(out, exist_ok=True)
    spec = dict(agent=os.path.relpath(agent, ROOT), opp={k: v for k, v in os.environ.items() if k.startswith("CAT")})
    meta = os.path.join(out, "arm.json")
    if os.path.exists(meta):
        assert json.load(open(meta)) == spec, f"{out} already ran with other settings"
    json.dump(spec, open(meta, "w"), indent=1)
    jobs = []
    for lo in range(0, a.games, a.chunk):
        f = os.path.join(out, f"{a.seed0}_{lo}_{min(lo + a.chunk, a.games)}.json")
        if not os.path.exists(f):
            jobs.append((agent, a.seed0, lo, min(lo + a.chunk, a.games), f))
    with Pool(a.par) as pool:
        for k, f in enumerate(pool.imap_unordered(play, jobs), 1):
            print(f"{k}/{len(jobs)} chunks done", flush=True)
    g = [r for f in os.listdir(out) if f.endswith(".json") and f != "arm.json"
         for r in json.load(open(os.path.join(out, f)))]
    w, n = sum(r["winner"] == "a" for r in g), len(g)
    p = w / n
    print(f"{os.path.basename(out):16s} {p:.3f} ± {1.96 * math.sqrt(p * (1 - p) / n):.3f} ({w}/{n})  "
          f"opponent {spec['opp']}", flush=True)


if __name__ == "__main__":
    main()
