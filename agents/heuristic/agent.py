"""Plain heuristics bot (heur.py): linear evaluation + one-move expectimax, no network."""

import random

from heur import HeurPlayer


class Agent:
    def __init__(self):
        self.p = HeurPlayer(None)
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        game = obs.sample_world(self.rng)
        acts = list(playable_actions)
        self.p.color = obs.me
        return acts.index(self.p.decide(game, acts))
