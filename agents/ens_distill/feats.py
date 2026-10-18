"""Hand-built position features, computed from one seat's point of view.

Only information that seat can see is used: for the opponent, the NUMBER of unplayed development
cards (never their types), and its hidden victory points as an expectation over the unseen pile.
The same function builds the training data and runs inside the agent, so the two cannot drift.
"""
import numpy as np

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionPrompt

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
RIDX = {r: i for i, r in enumerate(RES)}
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
PROBA = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
PROBA = {k: v / 36.0 for k, v in PROBA.items()}
NEIGH = {n: tuple(STATIC_GRAPH.neighbors(n)) for n in STATIC_GRAPH.nodes}

_KEYS = {}


def keys(i):
    k = _KEYS.get(i)
    if k is None:
        p = f"P{i}_"
        k = dict(
            hand=[p + r + "_IN_HAND" for r in RES],
            dev=[p + d + "_IN_HAND" for d in DEVS],
            vp=p + "VICTORY_POINTS", avp=p + "ACTUAL_VICTORY_POINTS",
            sett_av=p + "SETTLEMENTS_AVAILABLE", city_av=p + "CITIES_AVAILABLE",
            road_av=p + "ROADS_AVAILABLE", lr=p + "LONGEST_ROAD_LENGTH",
            has_road=p + "HAS_ROAD", has_army=p + "HAS_ARMY", knights=p + "PLAYED_KNIGHT",
            rolled=p + "HAS_ROLLED",
        )
        _KEYS[i] = k
    return k


class MapInfo:
    """Static facts about one board, computed once per decision (the view is a fresh object)."""

    def __init__(self, cmap):
        self.node_tiles = {}
        for n, tiles in cmap.adjacent_tiles.items():
            self.node_tiles[n] = [(t, RIDX[t.resource], PROBA[t.number])
                                  for t in tiles if t.resource is not None]
        self.node_val = {}
        for n, lst in self.node_tiles.items():
            s = sum(p for _, _, p in lst)
            self.node_val[n] = s + 0.02 * len({r for _, r, _ in lst})
        self.node_port = {}
        for res, nodes in cmap.port_nodes.items():
            for n in nodes:
                self.node_port[n] = 5 if res is None else RIDX[res]
        tot = [0.0] * 5
        for t in cmap.land_tiles.values():
            if t.resource is not None:
                tot[RIDX[t.resource]] += PROBA[t.number]
        self.board_prod = tot


_MAP_CACHE = {}


def map_info(cmap):
    mi = _MAP_CACHE.get(id(cmap))
    if mi is None or mi[0] is not cmap:
        if len(_MAP_CACHE) > 8:
            _MAP_CACHE.clear()
        mi = (cmap, MapInfo(cmap))
        _MAP_CACHE[id(cmap)] = mi
    return mi[1]


def _player_block(st, board, mi, color, i, robber_tile, enemy_color):
    """Features of one player that do not depend on hidden information."""
    ps = st.player_state
    k = keys(i)
    bl = st.buildings_by_color[color]
    setts = bl.get("SETTLEMENT", [])
    cities = bl.get("CITY", [])
    prod = [0.0] * 5
    nominal = 0.0
    ports = [0] * 6
    for mult, nodes in ((1, setts), (2, cities)):
        for n in nodes:
            for t, r, p in mi.node_tiles[n]:
                nominal += mult * p
                if t is not robber_tile:
                    prod[r] += mult * p
            pr = mi.node_port.get(n)
            if pr is not None:
                ports[pr] = 1
    hand = [ps[h] for h in k["hand"]]
    tot = sum(hand)
    w, b, s, wh, o = hand
    # Expansion: buildable spots now, and one or two roads away.
    comps = board.connected_components.get(color, [])
    buildable = board.board_buildable_ids
    level0 = set()
    for c in comps:
        level0 |= c
    roads = board.roads
    blds = board.buildings
    sett_av = ps[k["sett_av"]]
    def best_and_count(nodes):
        vals = [mi.node_val[n] for n in nodes if n in buildable]
        return (max(vals) if vals else 0.0), len(vals)
    b0, c0 = best_and_count(level0)
    seen = set(level0)
    frontier = level0
    out = [b0, min(c0, 6)]
    for _ in range(2):
        nxt = set()
        for n in frontier:
            bd = blds.get(n)
            if bd is not None and bd[0] != color:
                continue
            for m in NEIGH[n]:
                if m in seen:
                    continue
                rc = roads.get((n, m))
                if rc is not None and rc != color:
                    continue
                mb = blds.get(m)
                if mb is not None and mb[0] != color:
                    continue
                nxt.add(m)
        seen |= nxt
        bb, cc = best_and_count(nxt)
        out += [bb, min(cc, 6)]
        frontier = nxt
    port_eff = sum(prod[r] for r in range(5) if ports[r])
    feats = prod + [
        sum(prod), nominal, sum(1 for x in prod if x > 0),
        len(setts), len(cities), 15 - ps[k["road_av"]], float(sett_av == 0), float(ps[k["city_av"]] == 0),
        ps[k["lr"]], float(ps[k["has_road"]]), ps[k["knights"]], float(ps[k["has_army"]]),
    ] + [min(x, 8) for x in hand] + [
        tot, max(0, tot - 7), float(tot > 7),
        float(w >= 1 and b >= 1 and s >= 1 and wh >= 1),
        float(wh >= 2 and o >= 3),
        float(s >= 1 and wh >= 1 and o >= 1),
        float(w >= 1 and b >= 1),
        float(ports[5]), sum(ports[:5]), port_eff,
    ] + out
    return feats


def features(game, me):
    """Feature vector of `game` as seen by `me`. Uses no hidden information."""
    st = game.state
    board = st.board
    mi = map_info(board.map)
    i = st.color_to_index[me]
    j = 1 - i
    opp = st.colors[j]
    ps = st.player_state
    ki, kj = keys(i), keys(j)
    robber_tile = board.map.land_tiles[board.robber_coordinate]

    mine = _player_block(st, board, mi, me, i, robber_tile, opp)
    theirs = _player_block(st, board, mi, opp, j, robber_tile, me)

    my_devs = [ps[d] for d in ki["dev"]]
    opp_dev_n = sum(ps[d] for d in kj["dev"])
    unseen = len(st.development_listdeck) + opp_dev_n
    vp_left = 5 - my_devs[1]
    p_vp = vp_left / unseen if unseen > 0 else 0.0
    my_vp = ps[ki["avp"]]
    opp_pub = ps[kj["vp"]]
    opp_est = opp_pub + opp_dev_n * p_vp

    cur = st.colors[st.current_turn_index]
    my_turn = float(cur == me)
    prompt = st.current_prompt
    g = [
        my_vp, ps[ki["vp"]], opp_pub, opp_est, my_vp - opp_est,
        float(my_vp >= 8), float(my_vp >= 9), float(opp_est >= 8), float(opp_est >= 9),
        my_devs[0], my_devs[1], my_devs[2] + my_devs[3] + my_devs[4], opp_dev_n,
        len(st.development_listdeck), my_turn,
        float(ps[keys(st.current_turn_index)["rolled"]]),
        float(prompt == ActionPrompt.MOVE_ROBBER), float(prompt == ActionPrompt.DISCARD),
        float(st.is_initial_build_phase), min(st.num_turns, 200) / 100.0,
        sum(st.resource_freqdeck),
    ] + mi.board_prod
    return g + mine + theirs


N_FEATURES = None


def n_features():
    global N_FEATURES
    if N_FEATURES is None:
        from catanatron.game import Game
        from catanatron.models.player import Color, RandomPlayer
        gm = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
        N_FEATURES = len(features(gm, Color.RED))
    return N_FEATURES
