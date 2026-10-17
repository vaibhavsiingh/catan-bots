"""State/action encoding for the entity-token transformer.

Every board object becomes a token, seen from the perspective of the player
deciding (relative seating: me, next, next+1, ...):
  19 tiles | 54 nodes | 72 edges | 4 players | 1 global  -> 150 tokens
Each legal action is encoded as (type, pointer token, pointer2 token, resource vec).
"""

import numpy as np
from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.models.board import get_edges
from catanatron.models.enums import ActionPrompt

N_TILES, N_NODES, N_EDGES, N_PLAYERS = 19, 54, 72, 4
OFF_TILE = 0
OFF_NODE = OFF_TILE + N_TILES
OFF_EDGE = OFF_NODE + N_NODES
OFF_PLAYER = OFF_EDGE + N_EDGES
OFF_GLOBAL = OFF_PLAYER + N_PLAYERS
N_TOKENS = OFF_GLOBAL + 1

TILE_F, NODE_F, EDGE_F, PLAYER_F = 8, 21, 6, 36
PROMPTS = list(ActionPrompt)
GLOBAL_F = len(PROMPTS) + 12
ACTION_TYPES = list(ActionType)
ACT_RES_F = 10  # give(5) + get(5)
RES_IDX = {r: i for i, r in enumerate(RESOURCES)}
DEV_TYPES = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]
PORT_IDX = {None: 0, **{r: i + 1 for i, r in enumerate(RESOURCES)}}

_static_cache = {}


def _static(catan_map):
    """Per-map features that never change during a game."""
    key = id(catan_map)
    if key in _static_cache:
        return _static_cache[key]
    edges = [tuple(sorted(e)) for e in get_edges(catan_map.land_nodes)]
    edge_idx = {e: i for i, e in enumerate(edges)}
    coord_to_tile = {c: t.id for c, t in catan_map.land_tiles.items()}
    tile = np.zeros((N_TILES, TILE_F), np.float32)
    for c, t in catan_map.land_tiles.items():
        tile[t.id, 0 if t.resource is None else RES_IDX[t.resource] + 1] = 1
        if t.number:
            tile[t.id, 6] = (6 - abs(7 - t.number)) / 5.0
    node = np.zeros((N_NODES, NODE_F), np.float32)
    for res, nodes in catan_map.port_nodes.items():
        for n in nodes:
            node[n, 8 + PORT_IDX[res]] = 1
    for n in range(N_NODES):
        for r, p in catan_map.node_production[n].items():
            node[n, 14 + RES_IDX[r]] = p * 6
    node_tiles = {
        n: [t.id for t in catan_map.adjacent_tiles[n]] for n in range(N_NODES)
    }
    s = dict(edges=edges, edge_idx=edge_idx, coord_to_tile=coord_to_tile,
             tile=tile, node=node, node_tiles=node_tiles)
    if len(_static_cache) > 64:
        _static_cache.clear()
    _static_cache[key] = s
    return s


def rel_order(state, color):
    """Colors in seating order starting from `color`."""
    colors = state.colors
    i = colors.index(color)
    return [colors[(i + k) % len(colors)] for k in range(len(colors))]


def encode_state(game, color):
    state = game.state
    board = state.board
    st = _static(board.map)
    order = rel_order(state, color)
    rel = {c: i for i, c in enumerate(order)}
    ps = state.player_state

    tile = st["tile"].copy()
    robber_tile = st["coord_to_tile"][board.robber_coordinate]
    tile[robber_tile, 7] = 1

    node = st["node"].copy()
    for n, (c, btype) in board.buildings.items():
        node[n, (0 if btype == SETTLEMENT else 4) + rel[c]] = 1
    for n in st["node_tiles"]:
        if robber_tile in st["node_tiles"][n]:
            node[n, 19] = 1
    for n in board.buildable_node_ids(color):
        node[n, 20] = 1

    edge = np.zeros((N_EDGES, EDGE_F), np.float32)
    eidx = st["edge_idx"]
    for e, c in board.roads.items():
        if e[0] < e[1]:
            edge[eidx[e], rel[c]] = 1
    for e in board.buildable_edges(color):
        edge[eidx[tuple(sorted(e))], 4] = 1
    # longest road owner marker on its edges is overkill; flag "any road" instead
    edge[:, 5] = edge[:, :4].sum(1)

    player = np.zeros((N_PLAYERS, PLAYER_F), np.float32)
    for c, r in rel.items():
        k = f"P{state.color_to_index[c]}_"
        f = player[r]
        f[0] = 1
        f[1] = r == 0
        f[2] = ps[k + "VICTORY_POINTS"] / 10
        f[3] = ps[k + "ACTUAL_VICTORY_POINTS"] / 10 if r == 0 else 0
        hand = [ps[k + res + "_IN_HAND"] for res in RESOURCES]
        if r == 0:
            f[4:9] = np.array(hand) / 5
        f[9] = sum(hand) / 10
        devs = [ps[k + d + "_IN_HAND"] for d in DEV_TYPES]
        if r == 0:
            f[10:15] = devs
        f[15] = sum(devs) / 3
        f[16:21] = [ps[k + "PLAYED_" + d] for d in DEV_TYPES]
        f[16] /= 3
        f[21] = ps[k + "LONGEST_ROAD_LENGTH"] / 10
        f[22] = ps[k + "HAS_ROAD"]
        f[23] = ps[k + "HAS_ARMY"]
        f[24] = ps[k + "SETTLEMENTS_AVAILABLE"] / 5
        f[25] = ps[k + "CITIES_AVAILABLE"] / 4
        f[26] = ps[k + "ROADS_AVAILABLE"] / 15
        f[27] = ps[k + "HAS_ROLLED"]
        f[28] = ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
        for pr in board.get_player_port_resources(c):
            f[29 + PORT_IDX[pr]] = 1
        f[35] = c == state.current_color()

    g = np.zeros(GLOBAL_F, np.float32)
    g[PROMPTS.index(state.current_prompt)] = 1
    o = len(PROMPTS)
    g[o + 0] = state.is_initial_build_phase
    g[o + 1] = state.is_discarding
    g[o + 2] = state.is_moving_knight
    g[o + 3] = state.is_road_building
    g[o + 4] = state.free_roads_available / 2
    g[o + 5] = min(state.num_turns, 300) / 100
    g[o + 6] = len(state.development_listdeck) / 25
    g[o + 7:o + 12] = np.array(state.resource_freqdeck[:5]) / 19
    return tile, node, edge, player, g


def encode_actions(game, color, actions):
    """Returns (A,3) int array [type, ptr, ptr2] and (A,10) float array."""
    st = _static(game.state.board.map)
    rel = {c: i for i, c in enumerate(rel_order(game.state, color))}
    ints = np.full((len(actions), 3), -1, np.int64)
    res = np.zeros((len(actions), ACT_RES_F), np.float32)
    for i, a in enumerate(actions):
        t, v = a.action_type, a.value
        ints[i, 0] = ACTION_TYPES.index(t)
        if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            ints[i, 1] = OFF_NODE + v
        elif t == ActionType.BUILD_ROAD:
            ints[i, 1] = OFF_EDGE + st["edge_idx"][tuple(sorted(v))]
        elif t == ActionType.MOVE_ROBBER:
            ints[i, 1] = OFF_TILE + st["coord_to_tile"][v[0]]
            if v[1] is not None:
                ints[i, 2] = OFF_PLAYER + rel[v[1]]
        elif t == ActionType.DISCARD_RESOURCE:
            res[i, RES_IDX[v]] = 1
        elif t == ActionType.PLAY_MONOPOLY:
            res[i, 5 + RES_IDX[v]] = 1
        elif t == ActionType.PLAY_YEAR_OF_PLENTY:
            for r in v:
                res[i, 5 + RES_IDX[r]] += 1
        elif t == ActionType.MARITIME_TRADE:
            for r in v[:-1]:
                if r is not None:
                    res[i, RES_IDX[r]] += 0.25
            res[i, 5 + RES_IDX[v[-1]]] = 1
    return ints, res
