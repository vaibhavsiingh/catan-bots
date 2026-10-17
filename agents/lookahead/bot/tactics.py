"""Tactical moves the search must always consider (config `force`, off in the final bot).

The policy learned from ValueFunction, which barely values Longest Road, so a road that takes it
can rank below the search's top K, or the policy can be confident enough to skip the search. In
50 games of the final-v1 bot (without this), such a chance was neither searched nor taken by the
end of the turn 31 times for Longest Road, 3 times for an immediate win, 1 time for Largest Army.

forced_moves() finds, on game copies, within a time cap:
  * a settlement / city / knight / 1-2 road sequence that wins the game now
  * road 1 of a 1-2 road sequence (bought, or via Road Building) that takes Longest Road this turn
  * a knight that takes Largest Army
It returns the first moves, plus the road-2 follow-ups that complete a sequence (for depth2).
"""
import random
import time

from catanatron.models.enums import ActionType as T
from catanatron.state_functions import player_key


def _after(game, action):
    g = game.copy()
    rng = random.Random(0)                         # never advance the real game's dice generator
    g.random = rng
    g.state.random = rng
    g.execute(action)
    return g


def forced_moves(game, color, cap_ms=50.0):
    """(first moves, {first move: follow-ups}) - both may be empty. Stops searching after cap_ms."""
    deadline = time.perf_counter() + cap_ms / 1000
    key = player_key(game.state, color)
    had_road = game.state.player_state[key + "_HAS_ROAD"]
    had_army = game.state.player_state[key + "_HAS_ARMY"]
    first, follow = set(), {}

    def length(g):
        return g.state.player_state[key + "_LONGEST_ROAD_LENGTH"]

    def gains(g):
        return g.winning_color() == color or (not had_road and g.state.player_state[key + "_HAS_ROAD"])

    def ours(g):
        return g.winning_color() is None and g.state.current_color() == color

    def completing_roads(g):
        """Roads at g that lengthen our road and win or take Longest Road."""
        out, base = [], length(g)
        for r in g.playable_actions:
            if r.action_type == T.BUILD_ROAD and time.perf_counter() < deadline:
                g2 = _after(g, r)
                if length(g2) > base and gains(g2):
                    out.append(r)
        return out

    for a in game.playable_actions:
        if time.perf_counter() > deadline:
            break
        t = a.action_type
        if t in (T.BUILD_SETTLEMENT, T.BUILD_CITY):
            if _after(game, a).winning_color() == color:
                first.add(a)
        elif t == T.PLAY_KNIGHT_CARD:
            g2 = _after(game, a)
            if g2.winning_color() == color or (not had_army and g2.state.player_state[key + "_HAS_ARMY"]):
                first.add(a)
        elif t == T.BUILD_ROAD:
            g2 = _after(game, a)
            if length(g2) <= length(game):
                continue                            # doesn't lengthen our longest road
            if gains(g2):
                first.add(a)
            elif ours(g2):
                done = completing_roads(g2)
                if done:
                    first.add(a)
                    follow[a] = set(done)
        elif t == T.PLAY_ROAD_BUILDING:
            g2 = _after(game, a)                    # two free roads follow
            if completing_roads(g2):
                first.add(a)
            else:
                for r in g2.playable_actions:
                    if r.action_type == T.BUILD_ROAD and time.perf_counter() < deadline:
                        g3 = _after(g2, r)
                        if length(g3) > length(g2) and ours(g3) and completing_roads(g3):
                            first.add(a)
                            break
    return first, follow
