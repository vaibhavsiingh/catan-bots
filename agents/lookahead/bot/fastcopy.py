"""Faster, exactly equivalent copies of catanatron's Board and State (config fast_copy, off by default).

Our search spends most of its time in simulations, and most of that inside ValueFunction's decide(),
which copies the game once per legal move. catanatron's copies deep-copy two board caches and round-
trip two structures through pickle. Their contents are only ints, int tuples, strings and enums
(immutable), so copying the containers is enough. The containers keep their exact types, since two of
them are defaultdicts whose behaviour code relies on. Same results, less time; more simulations then
fit into the same search budget. Verified by tests/test_fastcopy.py and the identical-moves test.
"""
from collections import defaultdict

from catanatron.models.board import Board
from catanatron.state import State

_ORIGINAL = {"board": Board.copy, "state": State.copy}


def _dict_of(d, inner):
    """Copy a (default)dict, copying each value with inner(), keeping the container type."""
    out = defaultdict(d.default_factory) if isinstance(d, defaultdict) else {}
    for k, v in d.items():
        out[k] = inner(v)
    return out


def _list_copy(v):
    return list(v)


def _set_copy(v):
    return set(v)


def _sets_copy(v):          # list of sets
    return [set(x) for x in v]


def _lists_by_type(v):      # (default)dict building type -> list of node ids
    return _dict_of(v, _list_copy)


def board_copy(self):
    board = Board(self.map, initialize=False)
    board.map = self.map
    board.buildings = self.buildings.copy()
    board.roads = self.roads.copy()
    board.connected_components = _dict_of(self.connected_components, _sets_copy)
    board.board_buildable_ids = self.board_buildable_ids.copy()
    board.road_lengths = self.road_lengths.copy()
    board.road_color = self.road_color
    board.road_length = self.road_length
    board.robber_coordinate = self.robber_coordinate
    board.buildable_subgraph = self.buildable_subgraph
    board.buildable_edges_cache = _dict_of(self.buildable_edges_cache, _list_copy)
    board.player_port_resources_cache = _dict_of(self.player_port_resources_cache, _set_copy)
    return board


def state_copy(self):
    s = State([], None, initialize=False)
    s.random = self.random
    s.players = self.players
    s.discard_limit = self.discard_limit
    s.friendly_robber = self.friendly_robber
    s.board = self.board.copy()
    s.player_state = self.player_state.copy()
    s.color_to_index = self.color_to_index
    s.colors = self.colors
    s.resource_freqdeck = self.resource_freqdeck.copy()
    s.development_listdeck = self.development_listdeck.copy()
    s.buildings_by_color = _dict_of(self.buildings_by_color, _lists_by_type)
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


def install():
    Board.copy = board_copy
    State.copy = state_copy


def uninstall():
    Board.copy = _ORIGINAL["board"]
    State.copy = _ORIGINAL["state"]
