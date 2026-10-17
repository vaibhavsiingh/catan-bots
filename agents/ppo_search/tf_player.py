"""Catanatron player driven by the transformer policy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from catanatron.models.player import Player
from encode import encode_state, encode_actions
from model import CatanTransformer

_models = {}


def load_model(path):
    if path not in _models:
        ck = torch.load(path)
        m = CatanTransformer(d=ck["d"], layers=ck["layers"])
        m.load_state_dict(ck["model"])
        m.eval()
        _models[path] = m
    return _models[path]


def policy_inputs(game, color, actions):
    s = encode_state(game, color)
    ai, ar = encode_actions(game, color, actions)
    t = lambda x: torch.from_numpy(x).unsqueeze(0)
    return (*[t(x) for x in s], t(ai), t(ar), torch.ones(1, len(actions), dtype=torch.bool))


class TransformerPlayer(Player):
    """Greedy (or sampled, with temperature) policy from a trained checkpoint."""

    def __init__(self, color, ckpt=None, temperature=0.0):
        super().__init__(color)
        self.model = load_model(ckpt or os.environ.get("TF_CKPT", "ckpt.pt"))
        self.temperature = temperature

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        acts = list(playable_actions)
        with torch.no_grad():
            logits, _ = self.model(*policy_inputs(game, self.color, acts))
        if self.temperature > 0:
            i = torch.multinomial(torch.softmax(logits[0] / self.temperature, 0), 1).item()
        else:
            i = logits[0].argmax().item()
        return acts[i]
