"""PPO v2 transformer policy, greedy, one forward pass per decision."""

import os
import random

import torch

from encode import encode_state, encode_actions
from model import CatanTransformer

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))


class Agent:
    def __init__(self):
        ck = torch.load(os.path.join(HERE, "ppo2_it20.pt"), map_location="cpu")
        self.model = CatanTransformer(d=ck["d"], layers=ck["layers"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        game = obs.sample_world(self.rng)
        acts = list(playable_actions)
        s = encode_state(game, obs.me)
        ai, ar = encode_actions(game, obs.me, acts)
        t = lambda x: torch.from_numpy(x).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = self.model(*[t(x) for x in s], t(ai), t(ar), torch.ones(1, len(acts), dtype=torch.bool))
        return int(logits[0].argmax())
