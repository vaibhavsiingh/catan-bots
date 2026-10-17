"""Any of catanatron's own bots as a harness Agent, for a gauntlet with tools/ab.py.

env CAT picks the bot, with catanatron-play's codes: R (random), W (weighted random), VP (victory points),
G (greedy playouts, CAT_N playouts, default 25), M (MCTS, CAT_N simulations, default 100), F (value function),
AB (alpha-beta, CAT_DEPTH default 2, CAT_PRUNE default false), PY (any catanatron Player class:
CAT_CLASS=package.module:Class, importable from PYTHONPATH).
The bot searches a world sampled from the observation (the harness sends agents a masked game), so it does not
see the opponent's development cards; tools/native_match.py plays it with the true game. If its action is not among the playable ones, F's one-move choice is played."""

import importlib
import os
import random

from catanatron.models.player import RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players import playouts
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, ValueFunctionPlayer, base_fn
from catanatron.players.weighted_random import WeightedRandomPlayer

VALUE_FN = base_fn(DEFAULT_WEIGHTS)
playouts.USE_MULTIPROCESSING = False  # harness agents are daemon processes: a Pool there raises on every decision
CAT = os.environ.get("CAT", "AB")
N = os.environ.get("CAT_N")


def make(color):
    if CAT == "R":
        return RandomPlayer(color)
    if CAT == "W":
        return WeightedRandomPlayer(color)
    if CAT == "VP":
        return VictoryPointPlayer(color)
    if CAT == "G":
        return GreedyPlayoutsPlayer(color, int(N or 25))
    if CAT == "M":
        return MCTSPlayer(color, int(N or 100))
    if CAT == "F":
        return ValueFunctionPlayer(color)
    if CAT == "AB":
        return AlphaBetaPlayer(color, depth=int(os.environ.get("CAT_DEPTH", "2")),
                               prunning=os.environ.get("CAT_PRUNE", "false"))
    if CAT == "PY":
        module, _, cls = os.environ["CAT_CLASS"].partition(":")
        return getattr(importlib.import_module(module), cls)(color)
    raise ValueError(f"unknown CAT={CAT}")


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        random.seed(0)  # catanatron's random and playout bots use the global generator
        self.player = None

    def decide(self, obs, playable_actions) -> int:
        if len(playable_actions) == 1:
            return 0
        if self.player is None or self.player.color != obs.me:
            self.player = make(obs.me)
        world = obs.sample_world(self.rng)
        try:
            action = self.player.decide(world, world.playable_actions)
            for i, a in enumerate(playable_actions):
                if a == action:
                    return i
        except Exception:
            pass
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = VALUE_FN(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
