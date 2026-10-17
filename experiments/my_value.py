"""Our ValueFunction variant: same greedy 1-ply search, weights loaded from my_weights.json (tuned by tune.py)."""

import json
import os

from catanatron.players.value import ValueFunctionPlayer, base_fn, DEFAULT_WEIGHTS

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_weights.json")


def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            return {**DEFAULT_WEIGHTS, **json.load(f)}
    return dict(DEFAULT_WEIGHTS)


class MyValuePlayer(ValueFunctionPlayer):
    def __init__(self, color, weights=None):
        super().__init__(color)
        self.fn = base_fn(weights or load_weights())

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        best_value, best_action = float("-inf"), None
        for action in playable_actions:
            g = game.copy()
            g.execute(action)
            v = self.fn(g, self.color)
            if v > best_value:
                best_value, best_action = v, action
        return best_action
