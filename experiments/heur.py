"""Plain heuristics bot: a linear evaluation over hand-designed features, in victory-point units, and a
one-move look-ahead that averages over chance (dice, dev-card draws, robber steals). No network.

Features follow the V2 heuristic, and the starting weights are its CMA-ES-tuned V2-Tuned
weights; the code runs on both current catanatron and the pinned build in requirements.txt.

HeurPlayer(color, weights) is a catanatron player; Agent is the harness interface.
"""

import json
import os

from catanatron.models.enums import ActionPrompt, ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.state_functions import get_player_freqdeck, player_key
from catanatron.features import build_production_features, reachability_features

HERE = os.path.dirname(os.path.abspath(__file__))
_PROD = build_production_features(True)  # robber-aware ("effective") production

# V2-Tuned weights, rounded
WEIGHTS = dict(
    vp=9.54, opp_vp=-0.42,
    p_wood=1.38, p_brick=1.35, p_sheep=1.71, p_wheat=1.97, p_ore=3.98,
    variety=0.34, opp_prod=-5.19, reach0=2.97, reach1=0.82,
    spots=0.15, best0=1.49, best1=0.79,
    road_len=0.02, lr_lead=0.07, knights=0.27, army_lead=0.23,
    devs=0.18, knight_hand=0.07,
    city_need=-0.26, settle_need=-0.76, dev_need=-0.11, hand=0.05, discard=-0.31,
    port2=0.71, port3=0.18,
)


def features(game, me):
    st = game.state
    ps = st.player_state
    opp = next(c for c in st.colors if c != me)
    k, ok = player_key(st, me), player_key(st, opp)
    f = {"vp": ps[f"{k}_ACTUAL_VICTORY_POINTS"], "opp_vp": ps[f"{ok}_VICTORY_POINTS"]}
    pr = _PROD(game, me)  # P0 = me, P1 = opponent
    mine = [pr[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES]
    for r, v in zip(RESOURCES, mine):
        f["p_" + r.lower()] = v
    f["variety"] = sum(v > 0 for v in mine)
    f["opp_prod"] = sum(pr[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RESOURCES)
    rf = reachability_features(game, me, 2)  # production reachable with 0 / 1 more roads
    f["reach0"] = sum(rf[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    f["reach1"] = sum(rf[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

    board = st.board
    npd = board.map.node_production
    if ps[f"{k}_SETTLEMENTS_AVAILABLE"] > 0:
        spots = board.buildable_node_ids(me)
        f["spots"] = len(spots)
        f["best0"] = max((sum(npd[n].values()) for n in spots), default=0.0)
        one_road = set()
        if ps[f"{k}_ROADS_AVAILABLE"] > 0:
            free = board.board_buildable_ids
            for e in board.buildable_edges(me):
                one_road.update(n for n in e if n in free)
        f["best1"] = max((sum(npd[n].values()) for n in one_road), default=0.0)
    else:
        f["spots"] = f["best0"] = f["best1"] = 0.0

    my_lr, op_lr = ps[f"{k}_LONGEST_ROAD_LENGTH"], ps[f"{ok}_LONGEST_ROAD_LENGTH"]
    f["road_len"] = min(my_lr, 10)
    f["lr_lead"] = max(-3, min(3, my_lr - op_lr)) if max(my_lr, op_lr) >= 3 else 0
    my_kn, op_kn = ps[f"{k}_PLAYED_KNIGHT"], ps[f"{ok}_PLAYED_KNIGHT"]
    f["knights"] = min(my_kn, 6)
    f["army_lead"] = max(-3, min(3, my_kn - op_kn)) if max(my_kn, op_kn) >= 1 else 0
    f["devs"] = sum(ps[f"{k}_{d}_IN_HAND"] for d in ("KNIGHT", "MONOPOLY", "ROAD_BUILDING", "YEAR_OF_PLENTY"))
    f["knight_hand"] = ps[f"{k}_KNIGHT_IN_HAND"]

    h = get_player_freqdeck(st, me)  # wood brick sheep wheat ore
    f["city_need"] = (max(2 - h[3], 0) + max(3 - h[4], 0)) / 5
    f["settle_need"] = sum(max(1 - h[i], 0) for i in range(4)) / 4
    f["dev_need"] = (max(1 - h[2], 0) + max(1 - h[3], 0) + max(1 - h[4], 0)) / 3
    n = sum(h)
    f["hand"] = min(n, 10)
    f["discard"] = float(n > 7)
    ports = board.get_player_port_resources(me)
    f["port2"] = sum(v for r, v in zip(RESOURCES, mine) if r in ports)
    f["port3"] = float(None in ports)
    return f


def value(game, me, w):
    if game.winning_color() == me:
        return 1e6
    f = features(game, me)
    return sum(w[key] * f[key] for key in w)


def expected(game, action, me, w):
    return sum(p * value(g, me, w) for g, p in execute_spectrum(game, action))


class HeurPlayer(Player):
    def __init__(self, color, weights=None):
        super().__init__(color)
        self.w = dict(weights or WEIGHTS)
        self._turn, self._n = None, 0

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if game.state.current_prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            return next((a for a in playable_actions
                         if a.action_type in (ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE)), playable_actions[0])
        # guard against trade loops: after 40 decisions in one turn, roll or end the turn
        t = game.state.num_turns
        if t != self._turn:
            self._turn, self._n = t, 0
        self._n += 1
        if self._n > 40:
            for a in playable_actions:
                if a.action_type in (ActionType.END_TURN, ActionType.ROLL):
                    return a
        best, best_v = playable_actions[0], -float("inf")
        for a in playable_actions:
            v = expected(game, a, self.color, self.w)
            if v > best_v:
                best, best_v = a, v
        return best


def load(path):
    w = dict(WEIGHTS)
    if path:
        w.update(json.load(open(path)))
    return w
