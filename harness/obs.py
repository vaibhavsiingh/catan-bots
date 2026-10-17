"""What your agent sees, and how to search under imperfect information.

WHAT YOUR AGENT SEES
    the board, your own hand, the bank, the opponent's public victory points, the opponent's
    resource hand, the NUMBER of unplayed development cards the opponent holds, every
    development card either player has played, and the turn number.

WHAT IT DOES NOT SEE
    which development cards the opponent holds unplayed, and therefore the opponent's hidden
    victory points. The opponent can win on a turn that looks safe to you.

WHY THE OPPONENT'S RESOURCES ARE NOT HIDDEN
    There are 19 cards of each resource. The bank is public and your own hand is known, so the
    opponent's hand is exactly 19 - bank - mine for every resource. Hiding it would hide
    nothing. The development deck is the only real hidden information in 2-player Catan.

HOW TO SEARCH ANYWAY
    sample_world(rng) returns a complete catanatron Game that agrees with everything you can
    see, with the opponent's unplayed development cards drawn from the cards nobody has shown.
    Search that copy with the normal catanatron API. Sample several worlds and combine the
    results if you want to act well under uncertainty.
"""
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_TYPES = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
#: The complete development deck, 25 cards.
DEV_DECK = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2,
            "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}


class Obs:
    """A read-only view of the game from one seat. Do not mutate it."""

    def __init__(self, game, color):
        self._game = game
        self.me = color
        #: Seconds of CPU left in this game's allowance, this decision's increment included. Set by
        #: the harness; None when no clock applies. Your agent may use it to budget its search.
        self.cpu_left = None
        st = game.state
        i = st.color_to_index[color]
        j = 1 - i
        ps = st.player_state

        self.board = st.board
        self.turn = st.num_turns
        self.bank = {r: st.resource_freqdeck[k] for k, r in enumerate(RESOURCES)}

        self.my_resources = {r: ps[f"P{i}_{r}_IN_HAND"] for r in RESOURCES}
        self.my_devs = {d: ps[f"P{i}_{d}_IN_HAND"] for d in DEV_TYPES}
        self.my_public_vps = ps[f"P{i}_VICTORY_POINTS"]
        self.my_actual_vps = ps[f"P{i}_ACTUAL_VICTORY_POINTS"]

        # Exactly derivable from public information. See the module docstring.
        self.opp_resources = {r: ps[f"P{j}_{r}_IN_HAND"] for r in RESOURCES}
        # Hidden: you get the count, never the breakdown.
        self.opp_dev_count = sum(ps[f"P{j}_{d}_IN_HAND"] for d in DEV_TYPES)
        self.opp_public_vps = ps[f"P{j}_VICTORY_POINTS"]

        self.played_devs = {
            "me": {d: ps.get(f"P{i}_PLAYED_{d}", 0) for d in DEV_TYPES},
            "opp": {d: ps.get(f"P{j}_PLAYED_{d}", 0) for d in DEV_TYPES},
        }

    def sample_world(self, rng):
        """One complete Game consistent with this observation.

        From your seat the remaining deck and the opponent's unplayed development cards are a single
        unseen pile. This shuffles that pile, deals the opponent the number of cards it is known to
        hold, and leaves the rest as the deck. The observation's own game is not modified.
        """
        return masked_copy(self._game, self.me, rng)


def masked_copy(game, me, rng):
    """A copy of `game` in which everything `me` cannot see is replaced by a consistent random draw.

    The engine stores hidden information in four places, and all four must be covered. Masking only
    the hand fields leaves the rest readable:
      1. P<opp>_<CARD>_IN_HAND          the opponent's unplayed cards, by type
      2. P<opp>_<CARD>_OWNED_AT_START   the same types again, for the "not this turn" rule
      3. P<opp>_ACTUAL_VICTORY_POINTS   public points plus hidden victory-point cards
      4. action_records                  every BUY_DEVELOPMENT_CARD logs the card drawn, in both
                                         action.value and result, so the history alone rebuilds
                                         the opponent's whole hand
    Plus the deck order, which would reveal future draws. The engine reads only len(action_records),
    never its contents, so blanking the drawn card in those records cannot change how a game plays.
    """
    from catanatron.models.enums import Action, ActionRecord, ActionType

    g = game.copy()
    st = g.state
    j = 1 - st.color_to_index[me]
    opp = st.colors[j]
    ps = st.player_state

    unseen = list(st.development_listdeck)
    for d in DEV_TYPES:
        unseen += [d] * ps[f"P{j}_{d}_IN_HAND"]
    n_opp = sum(ps[f"P{j}_{d}_IN_HAND"] for d in DEV_TYPES)
    # SORT BEFORE SHUFFLING. Built as "true deck order, then the opponent's true cards", the pile
    # encodes the hidden truth in its order, and a shuffle with a reproducible seed is invertible:
    # undo it and the truth comes back. Sorted first, the only thing an inverse can recover is the
    # multiset, which is public anyway.
    unseen.sort()
    rng.shuffle(unseen)
    drawn, rest = unseen[:n_opp], unseen[n_opp:]
    for d in DEV_TYPES:
        c = drawn.count(d)
        ps[f"P{j}_{d}_IN_HAND"] = c
        if f"P{j}_{d}_OWNED_AT_START" in ps:
            ps[f"P{j}_{d}_OWNED_AT_START"] = c
    ps[f"P{j}_ACTUAL_VICTORY_POINTS"] = ps[f"P{j}_VICTORY_POINTS"] + drawn.count("VICTORY_POINT")
    st.development_listdeck = rest

    buy = ActionType.BUY_DEVELOPMENT_CARD
    st.action_records = [
        ActionRecord(Action(r.action.color, buy, None), None)
        if (r.action.action_type == buy and r.action.color == opp) else r
        for r in st.action_records]
    return g
