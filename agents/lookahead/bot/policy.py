"""Our network as Catanatron players: ModelPlayer (policy argmax) and PolicyPlayer (+ nudges).
The search player built on top of these is in bot/search.py; all settings are in bot/config.py."""
import numpy as np
import torch

from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Player
from bot import nudges
from bot.config import Config
from bot.model import load


def _parse_bias(text):
    out = {}
    for part in filter(None, (text or "").split(",")):
        name, value = part.split(":")
        out[name.strip()] = float(value)
    return out


def action_bias(bias, actions):
    """Optional logit bonus per action type (config `bias`, e.g. "BUY_DEVELOPMENT_CARD:2")."""
    if not bias:
        return 0.0
    return np.array([bias.get(a.action_type.value, 0.0) for a in actions])


class ModelPlayer(Player):
    """Picks the legal move with the highest score. Settings: cfg, or Config.load() (environment
    variables and CATAN_VARIANT, so parallel referee workers see them too); path overrides cfg.model."""

    def __init__(self, color, params=None, path=None, cfg=None):
        super().__init__(color, params)
        self.cfg = cfg or Config.load()
        self.path = path or self.cfg.model
        self.bias = _parse_bias(self.cfg.bias)
        torch.set_num_threads(1)                 # one thread per game; parallelism comes from processes

    def scores(self, game):
        model, enc, table = load(self.path, game.state.board.map)
        x = enc(game, self.color)
        idx, mask = table.legal(game.playable_actions)
        with torch.no_grad():
            logits, value = model({k: torch.from_numpy(v)[None] for k, v in x.items()},
                                  torch.from_numpy(mask)[None])
        return logits[0, idx].numpy(), torch.sigmoid(value[0]).item()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if game.state.current_prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            for a in playable_actions:          # never trade with the opponent
                if a.action_type in (ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE):
                    return a
            return playable_actions[0]
        s, _ = self.scores(game)
        return playable_actions[int(np.argmax(s + action_bias(self.bias, playable_actions)))]


class PolicyPlayer(ModelPlayer):
    """Policy argmax + the strategy nudges of the config (no search)."""

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1 or game.state.current_prompt in (
                ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            return super().decide(game, playable_actions)
        logits, _ = self.scores(game)
        bonus = nudges.total(self.cfg.nudges, game, self.color, playable_actions)
        return playable_actions[int(np.argmax(logits + bonus + action_bias(self.bias, playable_actions)))]
