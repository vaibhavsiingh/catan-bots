"""Is the opponent F? A check on its opening placements, which F chooses deterministically.

One agent process plays many games against the same opponent (the harness keeps it loaded), and
adapting to it is allowed. Our search models the opponent with F's own code, which is exact against
F and only a guess against anyone else. This detector replays the opening from the action log (it is
deterministic: no dice, no hidden cards) and asks, at every opponent placement, which move F would
have made there. Any mismatch means the opponent is not F.

Cost: about 4 F decisions per game (2 settlements, 2 roads), and only until `max_checks` placements
were compared or a mismatch was found.
"""
import random

from catanatron import Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)
OPENING = (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)


def f_choice(game, color):
    """The move F (examples/f_agent.py) plays here: the first action with the best successor value."""
    best, best_value = None, None
    for action in game.playable_actions:
        successor = game.copy()
        try:
            successor.execute(action)
        except Exception:
            continue
        value = VALUE_FN(successor, color)
        if best_value is None or value > best_value:
            best, best_value = action, value
    return best


def opening_replica(view):
    """A fresh game on the view's board, with the view's seat order, before any placement."""
    saved = random.getstate()                      # Game() reseeds the global generator
    try:
        g = Game([RandomPlayer(c) for c in view.state.colors], catan_map=view.state.board.map)
    finally:
        random.setstate(saved)
    st = g.state                                   # Game() shuffles the seats: put them back
    st.players = [RandomPlayer(c) for c in view.state.colors]
    st.colors = tuple(view.state.colors)
    st.color_to_index = {c: i for i, c in enumerate(st.colors)}
    g.playable_actions = generate_playable_actions(st)
    return g


class OppDetector:
    def __init__(self, max_checks=12):
        self.max_checks = max_checks
        self.checks = 0
        self.mismatches = 0
        self.next_record = 0           # first action record of this game not compared yet

    def new_game(self):
        self.next_record = 0

    @property
    def decided(self):
        return self.mismatches > 0 or self.checks >= self.max_checks

    @property
    def is_f(self):
        """True until a mismatch is seen: while unsure, the opponent model stays F."""
        return self.mismatches == 0

    def observe(self, view, me):
        """Compare every opponent opening placement in the log that wasn't compared yet."""
        if self.decided:
            return
        records = view.state.action_records
        n_open = next((k for k, r in enumerate(records) if r.action.action_type == ActionType.ROLL), len(records))
        opp = next(c for c in view.state.colors if c != me)
        todo = [k for k in range(self.next_record, n_open)
                if records[k].action.color == opp and records[k].action.action_type in OPENING]
        self.next_record = max(self.next_record, n_open)
        if not todo:
            return
        g = opening_replica(view)
        done = 0
        for k in todo:
            for r in records[done:k]:
                g.execute(r.action, validate_action=False)
            done = k
            choice = f_choice(g, opp)
            actual = records[k].action
            self.checks += 1
            if choice is None or (choice.action_type, choice.value) != (actual.action_type, actual.value):
                self.mismatches += 1
                return
            if self.decided:
                return
