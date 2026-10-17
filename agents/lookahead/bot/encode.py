"""Board topology, the fixed move table, and the state encoder (Catanatron 3.3.0, 1v1)."""
import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, ActionPrompt, ActionType
from catanatron.state_functions import player_key

class Topology:
    """Fixed facts about the board graph. Positions never change between games;
    only what is on the tiles (resource, number) changes."""

    def __init__(self, board_map):
        self.tile_coords = sorted(board_map.land_tiles)                        # 19 tiles
        self.tile_index = {c: i for i, c in enumerate(self.tile_coords)}
        self.nodes = sorted(board_map.land_nodes)                              # 54 corners
        assert self.nodes == list(range(54))
        self.edges = sorted(tuple(sorted(e)) for e in get_edges(board_map.land_nodes))  # 72 edges
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.tile_nodes = [sorted(board_map.land_tiles[c].nodes.values()) for c in self.tile_coords]
        self.node_tiles = [[] for _ in self.nodes]                             # tiles touching each node
        for t, ns in enumerate(self.tile_nodes):
            for n in ns:
                self.node_tiles[n].append(t)


GLOBAL_MOVES = (
    [("ROLL", None), ("END_TURN", None), ("BUY_DEVELOPMENT_CARD", None),
     ("PLAY_KNIGHT_CARD", None), ("PLAY_ROAD_BUILDING", None)]
    + [("DISCARD_RESOURCE", r) for r in RESOURCES]
    + [("PLAY_MONOPOLY", r) for r in RESOURCES]
    + [("PLAY_YEAR_OF_PLENTY", (RESOURCES[i], RESOURCES[j])) for i in range(5) for j in range(i, 5)]
    + [("PLAY_YEAR_OF_PLENTY", (r,)) for r in RESOURCES]
    + [("MARITIME_TRADE", tuple([give] * k + [None] * (4 - k) + [get]))
       for k in (4, 3, 2) for give in RESOURCES for get in RESOURCES if give != get]
)


class MoveTable:
    """Every move that can ever be legal, in a fixed order, laid out in blocks that line up
    with the tokens that will score them:
        [settle x54 | city x54 | road x72 | robber x19 | global x95]  = 294 moves"""

    def __init__(self, topo):
        self.keys = ([("BUILD_SETTLEMENT", n) for n in topo.nodes]
                     + [("BUILD_CITY", n) for n in topo.nodes]
                     + [("BUILD_ROAD", e) for e in topo.edges]
                     + [("MOVE_ROBBER", c) for c in topo.tile_coords]
                     + GLOBAL_MOVES)
        self.index = {k: i for i, k in enumerate(self.keys)}
        assert len(self.index) == len(self.keys), "duplicate keys"
        self.n = len(self.keys)
        self.SETTLE, self.CITY, self.ROAD = slice(0, 54), slice(54, 108), slice(108, 180)
        self.ROBBER, self.GLOBAL = slice(180, 199), slice(199, self.n)

    @staticmethod
    def key(action):
        """A description of a move that ignores colours and argument order."""
        t, v = action.action_type, action.value
        if t == ActionType.BUILD_ROAD:
            v = tuple(sorted(v))
        elif t == ActionType.MOVE_ROBBER:
            v = v[0]                                 # 1v1: the tile decides who is robbed
        elif t == ActionType.PLAY_YEAR_OF_PLENTY:
            v = tuple(sorted(v, key=RESOURCES.index))
        return (t.value, v)

    def legal(self, playable_actions):
        """Indices of the legal moves (same order as the list) and a boolean mask over the table."""
        idx = [self.index[self.key(a)] for a in playable_actions]
        mask = np.zeros(self.n, dtype=bool)
        mask[idx] = True
        return idx, mask


PROMPTS = list(ActionPrompt)
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
F_TILE, F_NODE, F_EDGE, F_ME, F_OPP, F_GLOBAL = 10, 19, 3, 32, 16, 17


def pips(number):
    return 0 if number is None else 6 - abs(7 - number)


class Encoder:
    """(game, colour to move) -> arrays of token features, always from the mover's point of view:
    columns say 'mine' and 'opponent's', never RED and BLUE. One network can then play both seats."""

    def __init__(self, topo, see_opponent_hand=False):
        self.topo = topo
        self.see_opponent_hand = see_opponent_hand
        self._static_for = None

    def _static(self, board_map):
        """Features fixed for the whole game (tile contents, node production, ports), computed once."""
        if self._static_for is board_map:
            return
        T = self.topo
        tile = np.zeros((19, F_TILE), np.float32)
        self.tile_pips = np.zeros((19, 5), np.float32)        # pips per resource on each tile
        for i, c in enumerate(T.tile_coords):
            lt = board_map.land_tiles[c]
            r = 5 if lt.resource is None else RESOURCES.index(lt.resource)
            tile[i, r] = 1                                      # cols 0-5: resource (5 = desert)
            tile[i, 6] = pips(lt.number) / 5                    # col 6: how often it is rolled
            if lt.resource is not None:
                self.tile_pips[i, r] = pips(lt.number)
        node = np.zeros((54, F_NODE), np.float32)
        for n in T.nodes:
            prod = self.tile_pips[T.node_tiles[n]].sum(0)
            node[n, 4:9] = prod / 5                             # cols 4-8: pips per resource here
            node[n, 9] = prod.sum() / 15                        # col 9: total pips here
        for resource, ns in board_map.port_nodes.items():
            col = 10 if resource is None else 11 + RESOURCES.index(resource)
            node[list(ns), col] = 1                             # cols 10-15: 3:1 port, 2:1 ports
        self.static_tile, self.static_node = tile, node
        self._static_for = board_map

    def player_production(self, state, colour):
        """Expected cards per 36 rolls, per resource: cities count twice, the robbed tile counts zero."""
        T, b = self.topo, state.board
        robbed = T.tile_index[b.robber_coordinate]
        out = np.zeros(5, np.float32)
        for n, (c, kind) in b.buildings.items():
            if c == colour:
                for t in T.node_tiles[n]:
                    if t != robbed:
                        out += (2 if kind == "CITY" else 1) * self.tile_pips[t]
        return out

    def __call__(self, game, me):
        s, b, T = game.state, game.state.board, self.topo
        self._static(b.map)
        opp = next(c for c in s.colors if c != me)
        ps, km, ko = s.player_state, player_key(s, me), player_key(s, opp)

        tile = self.static_tile.copy()
        robbed = T.tile_index[b.robber_coordinate]
        tile[robbed, 7] = 1                                     # col 7: robber here
        node = self.static_node.copy()
        for n, (c, kind) in b.buildings.items():
            node[n, (0 if kind == "SETTLEMENT" else 1) + (0 if c == me else 2)] = 1   # cols 0-3
            for t in T.node_tiles[n]:                           # cols 8-9 of tiles: who is on it
                tile[t, 8 if c == me else 9] += (2 if kind == "CITY" else 1) / 6
        initial = s.is_initial_build_phase
        node[b.buildable_node_ids(me, initial), 16] = 1         # col 16: I could settle here
        node[b.buildable_node_ids(opp, initial), 17] = 1        # col 17: opponent could settle here
        node[T.tile_nodes[robbed], 18] = 1                      # col 18: next to the robber

        edge = np.zeros((72, F_EDGE), np.float32)
        for e, c in b.roads.items():
            if e[0] < e[1]:                                     # roads are stored in both directions
                edge[T.edge_index[e], 0 if c == me else 1] = 1  # cols 0-1: my road, their road
        for e in b.buildable_edges(me):
            edge[T.edge_index[tuple(sorted(e))], 2] = 1         # col 2: I could build a road here

        hand = np.array([ps[f"{km}_{r}_IN_HAND"] for r in RESOURCES], np.float32)
        me_vec = np.concatenate([
            hand / 5, [hand.sum() / 10, float(hand.sum() > 7)],
            [ps[f"{km}_{d}_IN_HAND"] / 2 for d in DEVS],
            [float(ps[f"{km}_{d}_OWNED_AT_START"] > 0 and not ps[f"{km}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])
             for d in ["KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]],
            [ps[f"{km}_ACTUAL_VICTORY_POINTS"] / 10, ps[f"{km}_VICTORY_POINTS"] / 10,
             ps[f"{km}_PLAYED_KNIGHT"] / 3, ps[f"{km}_LONGEST_ROAD_LENGTH"] / 10,
             float(ps[f"{km}_HAS_ROAD"]), float(ps[f"{km}_HAS_ARMY"]),
             ps[f"{km}_ROADS_AVAILABLE"] / 15, ps[f"{km}_SETTLEMENTS_AVAILABLE"] / 5,
             ps[f"{km}_CITIES_AVAILABLE"] / 4,
             float(ps[f"{km}_HAS_ROLLED"]), float(ps[f"{km}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])],
            self.player_production(s, me) / 10,
        ]).astype(np.float32)

        ohand = np.array([ps[f"{ko}_{r}_IN_HAND"] for r in RESOURCES], np.float32)
        opp_vec = np.concatenate([
            [ohand.sum() / 10, float(ohand.sum() > 7), sum(ps[f"{ko}_{d}_IN_HAND"] for d in DEVS) / 5,
             ps[f"{ko}_VICTORY_POINTS"] / 10, ps[f"{ko}_PLAYED_KNIGHT"] / 3,
             ps[f"{ko}_LONGEST_ROAD_LENGTH"] / 10, float(ps[f"{ko}_HAS_ROAD"]), float(ps[f"{ko}_HAS_ARMY"]),
             ps[f"{ko}_ROADS_AVAILABLE"] / 15, ps[f"{ko}_SETTLEMENTS_AVAILABLE"] / 5,
             ps[f"{ko}_CITIES_AVAILABLE"] / 4],
            self.player_production(s, opp) / 10,
        ]).astype(np.float32)
        if self.see_opponent_hand:                              # optional: the exact opponent hand
            opp_vec = np.concatenate([opp_vec, ohand / 5]).astype(np.float32)

        glob = np.zeros(F_GLOBAL, np.float32)
        glob[PROMPTS.index(s.current_prompt)] = 1                # cols 0-6: what is being asked
        glob[7] = min(s.num_turns / 200, 1.5)                   # col 7: how far into the game
        glob[8:13] = np.array(s.resource_freqdeck, np.float32) / 19   # cols 8-12: bank
        glob[13] = len(s.development_listdeck) / 25             # col 13: dev cards left
        glob[14] = s.free_roads_available / 2                   # col 14: free roads (road building)
        glob[15] = s.discard_counts[s.color_to_index[me]] / 10  # col 15: cards I still must discard
        glob[16] = float(s.colors[0] == me)                     # col 16: I went first
        return {"tile": tile, "node": node, "edge": edge, "me": me_vec, "opp": opp_vec, "global": glob}
