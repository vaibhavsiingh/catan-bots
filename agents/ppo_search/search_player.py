"""Policy-guided 1-ply expectimax with the transformer's value head.

For the policy's top-k actions, enumerate chance outcomes (dice, dev-card draws, steals) with
catanatron's execute_spectrum, score every resulting state with the value head in ONE batched
forward pass, and pick argmax_a  E[V(s')] + beta * log pi(a).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import execute_spectrum
from encode import encode_state, OFF_GLOBAL
from tf_player import load_model, policy_inputs


def value_batch(model, games, color):
    states = [encode_state(g, color) for g in games]
    x = [torch.from_numpy(np.stack([s[i] for s in states])) for i in range(5)]
    with torch.no_grad():
        h = model.encode(*x)
        return torch.sigmoid(model.value(h[:, OFF_GLOBAL]).squeeze(-1)).numpy()


_vnets = {}


def load_vnet(path):
    if path not in _vnets:
        from value_model import ValueNet
        ck = torch.load(path, map_location="cpu")
        m = ValueNet(d=ck["d"], layers=ck["layers"]); m.load_state_dict(ck["model"]); m.eval()
        _vnets[path] = m
    return _vnets[path]


def vnet_batch(vnet, games, color, lam):
    states = [encode_state(g, color) for g in games]
    x = [torch.from_numpy(np.stack([s[i] for s in states])) for i in range(5)]
    with torch.no_grad():
        wl, dvp, _ = vnet(*x)
    return (torch.sigmoid(wl) + lam * dvp).numpy()


class SearchPlayer(Player):
    """vnet=None -> use the policy net's own value head; else a dedicated ValueNet with
    score = P(win) + lam * predicted short-horizon VP-diff change."""

    def __init__(self, color, ckpt, k=5, beta=0.05, vnet=None, lam=0.1):
        super().__init__(color)
        self.model = load_model(ckpt)
        self.vnet = load_vnet(vnet) if vnet else None
        self.k, self.beta, self.lam = k, beta, lam

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        acts = list(playable_actions)
        with torch.no_grad():
            logits, _ = self.model(*policy_inputs(game, self.color, acts))
        logp = torch.log_softmax(logits[0], 0).numpy()
        cand = np.argsort(-logp)[: self.k]
        games, owner, probs = [], [], []
        for j, i in enumerate(cand):
            for g, p in execute_spectrum(game, acts[i]):
                games.append(g); owner.append(j); probs.append(p)
        v = (vnet_batch(self.vnet, games, self.color, self.lam) if self.vnet
             else value_batch(self.model, games, self.color))
        ev = np.zeros(len(cand))
        np.add.at(ev, owner, v * np.array(probs))
        score = ev + self.beta * logp[cand]
        return acts[cand[int(np.argmax(score))]]
