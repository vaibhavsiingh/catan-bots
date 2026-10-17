"""F with other weights, as a harness Agent: agents/f/agent.py (one-move lookahead with catanatron's base_fn), where
the environment variable FW holds JSON overrides of DEFAULT_WEIGHTS, e.g. FW='{"production": 1e6}'.
For scanning F's weights with tools/ab.py."""

import json
import os
import random

from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

WEIGHTS = dict(DEFAULT_WEIGHTS, **json.loads(os.environ.get("FW", "{}")))
VALUE_FN = base_fn(WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
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
