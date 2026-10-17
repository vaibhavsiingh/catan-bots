"""Strategy nudges: small logit bonuses on top of the policy, aimed at the teacher's blind spots.

Each nudge is a function (game, color, actions) -> one bonus per action (a numpy array). It is
registered under a name with @nudge("name") and switched on by giving it a weight in the config:
    CATAN_ROAD=3            (environment variable)
    "road=3,army=2"         (variant string, see bot/config.py)
The bonus that reaches the policy is the sum of weight * nudge over all nudges with a weight.

Adding one (it is then testable as "mynudge=1" with arena/ab.py):

    @nudge("cities")
    def cities(game, color, actions):
        return np.array([1.0 if a.action_type == AT.BUILD_CITY else 0.0 for a in actions])

Measured (v5 policy alone vs ValueFunction, 600 games, baseline 53.0%):
    road=3: 55.3% and 54.7% on two seed sets; with search: 67.7% vs 66.0% (600 games). In the final bot.
    ows (ore-wheat-sheep opening): 50.3%.  ows + army: 51-52.5%.  mono=1: 52.3%.
"""
import numpy as np

from catanatron.models.enums import ActionType as AT
from catanatron.state_functions import player_key

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
REGISTRY = {}


def nudge(name):
    def register(fn):
        REGISTRY[name] = fn
        return fn
    return register


def total(weights, game, color, actions):
    """Sum of weight * nudge for every nudge with a non-zero weight."""
    bonus = np.zeros(len(actions))
    for name, weight in weights.items():
        if weight:
            bonus += weight * REGISTRY[name](game, color, actions)
    return bonus


def _keys(game, color):
    s = game.state
    return s.player_state, player_key(s, color), player_key(s, next(c for c in s.colors if c != color))


@nudge("road")
def road(game, color, actions):
    """1 for a road that lengthens our longest road while the Longest Road race is close."""
    ps, me, op = _keys(game, color)
    out = np.zeros(len(actions))
    mine, theirs = ps[me + "_LONGEST_ROAD_LENGTH"], ps[op + "_LONGEST_ROAD_LENGTH"]
    if not (mine >= 3 and mine + 2 >= theirs and ps[me + "_ROADS_AVAILABLE"] > 0):
        return out
    for i, a in enumerate(actions):
        if a.action_type == AT.BUILD_ROAD:
            trial = game.copy()
            trial.execute(a)
            if trial.state.player_state[me + "_LONGEST_ROAD_LENGTH"] > mine:
                out[i] = 1.0
    return out


@nudge("army")
def army(game, color, actions):
    """1 for buying a dev card (when no city/settlement is possible) and for playing a knight,
    while Largest Army is still open. Largest Army is claimed in only ~3% of games vs ValueFunction."""
    ps, me, op = _keys(game, color)
    out = np.zeros(len(actions))
    if ps[op + "_HAS_ARMY"] or ps[me + "_PLAYED_KNIGHT"] >= 3:
        return out
    types = [a.action_type for a in actions]
    can_build = AT.BUILD_CITY in types or AT.BUILD_SETTLEMENT in types
    for i, t in enumerate(types):
        if (t == AT.BUY_DEVELOPMENT_CARD and not can_build) or t == AT.PLAY_KNIGHT_CARD:
            out[i] = 1.0
    return out


@nudge("mono")
def mono(game, color, actions):
    """Play Monopoly only for a resource the opponent holds at least 3 of (use weight 1)."""
    s = game.state
    ps, me, _ = _keys(game, color)
    out = np.zeros(len(actions))
    for i, a in enumerate(actions):
        if a.action_type == AT.PLAY_MONOPOLY:
            held = 19 - s.resource_freqdeck[RESOURCES.index(a.value)] - ps[f"{me}_{a.value}_IN_HAND"]
            out[i] = (5.0 if held >= 3 else -20.0) + 0.5 * held
    return out


@nudge("ows")
def ows(game, color, actions):
    """Ore-wheat-sheep opening: (ore + wheat + 0.5 sheep pips) / 5 for opening settlements."""
    s = game.state
    out = np.zeros(len(actions))
    if not s.is_initial_build_phase:
        return out
    for i, a in enumerate(actions):
        if a.action_type == AT.BUILD_SETTLEMENT:
            prod = s.board.map.node_production[a.value]
            out[i] = 36 * (prod.get("ORE", 0) + prod.get("WHEAT", 0) + 0.5 * prod.get("SHEEP", 0)) / 5
    return out
