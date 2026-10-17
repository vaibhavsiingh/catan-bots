"""Make any in-process player decide from what a protocol bot is shown, not from the true game.

In-process (the referee, the web server) a player is handed the real Game: the opponent's dev cards
and the order of the dev deck are visible. The competition harness sends a redacted view instead.
Redacted(player) rebuilds a full game from that redacted view (bot/hidden_info.py: exact in 1v1 except
the deck order, which is shuffled) and lets the player decide there, so every bot - ours or anyone's -
plays with exactly the information it would have in the competition."""
import sys

from catanatron.models.player import Player
from catanatron.serialization import action_to_json, client_view, state_to_json

from bot.hidden_info import rebuild_game


def redacted_game(game, color):
    view = client_view(state_to_json(game), color)
    return rebuild_game(view, view["map"], color.value)


def decide_redacted(decide, game, playable_actions, color):
    """decide(game, actions) on the redacted rebuild; returns the real game's matching action."""
    if len(playable_actions) == 1:
        return playable_actions[0]
    seen = redacted_game(game, color)
    chosen = action_to_json(decide(seen, seen.playable_actions))
    for action in playable_actions:
        if action_to_json(action) == chosen:
            return action
    print(f"redacted: chosen action not legal in the real game: {chosen}", file=sys.stderr, flush=True)
    return playable_actions[0]


class Redacted(Player):
    """Wraps a catanatron player: same color, decides on the redacted view; other attributes are the
    wrapped player's. Observer hooks (before/step/after) are not forwarded: none of our bots use them."""

    def __init__(self, player):
        super().__init__(player.color, getattr(player, "params", None))
        self.player = player

    @property
    def IS_BOT(self):
        return self.player.is_bot()

    def decide(self, game, playable_actions):
        return decide_redacted(self.player.decide, game, playable_actions, self.player.color)

    def __getattr__(self, name):                 # only called for attributes the wrapper lacks
        if name == "player":
            raise AttributeError(name)
        return getattr(self.player, name)
