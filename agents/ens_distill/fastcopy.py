"""Faster Game.copy for search, installed in the agent's own process only.

catanatron copies a state with pickle round-trips and copy.deepcopy, which dominate the cost of
one search node. These replacements build the same containers directly: same types (defaultdicts
stay defaultdicts), same contents, no shared mutable objects. Checked by replaying full games with
and without the patch: identical moves and results (lab/check_fastcopy.py).
"""
from collections import defaultdict

from catanatron.models.board import Board
from catanatron.state import State

_APPLIED = False


def _board_copy(self):
    board = Board(self.map, initialize=False)
    board.map = self.map
    board.buildings = self.buildings.copy()
    board.roads = self.roads.copy()
    cc = defaultdict(list)
    for c, comps in self.connected_components.items():
        cc[c] = [set(s) for s in comps]
    board.connected_components = cc
    board.board_buildable_ids = self.board_buildable_ids.copy()
    board.road_lengths = self.road_lengths.copy()
    board.road_color = self.road_color
    board.road_length = self.road_length
    board.robber_coordinate = self.robber_coordinate
    board.buildable_subgraph = self.buildable_subgraph
    board.buildable_edges_cache = {c: list(v) for c, v in self.buildable_edges_cache.items()}
    board.player_port_resources_cache = {c: set(v) for c, v in self.player_port_resources_cache.items()}
    return board


def _state_copy(self):
    s = State([], None, initialize=False)
    s.players = self.players
    s.discard_limit = self.discard_limit
    s.friendly_robber = self.friendly_robber
    s.board = self.board.copy()
    s.player_state = self.player_state.copy()
    s.color_to_index = self.color_to_index
    s.colors = self.colors
    s.resource_freqdeck = self.resource_freqdeck.copy()
    s.development_listdeck = self.development_listdeck.copy()
    bbc = {}
    for c, d in self.buildings_by_color.items():
        nd = defaultdict(list)
        for k, v in d.items():
            nd[k] = list(v)
        bbc[c] = nd
    s.buildings_by_color = bbc
    s.action_records = self.action_records.copy()
    s.num_turns = self.num_turns
    s.current_player_index = self.current_player_index
    s.current_turn_index = self.current_turn_index
    s.current_prompt = self.current_prompt
    s.is_initial_build_phase = self.is_initial_build_phase
    s.is_discarding = self.is_discarding
    s.discard_counts = self.discard_counts.copy()
    s.is_moving_knight = self.is_moving_knight
    s.is_road_building = self.is_road_building
    s.free_roads_available = self.free_roads_available
    s.is_resolving_trade = self.is_resolving_trade
    s.current_trade = self.current_trade
    s.acceptees = self.acceptees
    return s


def apply():
    global _APPLIED
    if not _APPLIED:
        Board.copy = _board_copy
        State.copy = _state_copy
        _APPLIED = True
