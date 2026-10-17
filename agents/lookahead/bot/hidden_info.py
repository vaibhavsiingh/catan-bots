"""Rebuild a full Catanatron game from the redacted view a protocol bot receives.

What the view hides from us, and how we get it back (1v1 only):
  * Opponent's resource cards: every resource has 19 cards, each one in the bank (public),
    our hand (known) or the opponent's hand. So opponent = 19 - bank - ours. Exact.
  * Opponent's development cards: the deck's composition is public, our hand is known, and every
    played card is public. So opponent = initial deck - remaining deck - ours - all played. Exact.
  * Only the ORDER of the remaining deck and the random seed are truly unknown: we shuffle.
Verified exact on 355 positions from 30 ValueFunction games.
If the view is inconsistent or incomplete, or the game is not 1v1, we warn once on stderr and guess a
hand that matches the opponent's public card counts, instead of failing."""
import random
import sys

from catanatron import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color, RandomPlayer
from catanatron.serialization import state_from_json

INITIAL_DEV = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}
PLAYABLE = ["KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
CARDS_PER_RESOURCE = 19


def _split(total, weights):
    """Split an integer total in proportion to non-negative weights (largest remainder)."""
    w = [max(0, x) for x in weights]
    if total <= 0:
        return [0] * len(w)
    if sum(w) == 0:
        w = [1] * len(w)
    raw = [total * x / sum(w) for x in w]
    out = [int(r) for r in raw]
    for i in sorted(range(len(w)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


_warned = []


def _warn_once(message):
    if not _warned:
        print(f"hidden_info: {message}; using a guessed opponent hand (further occurrences not reported)",
              file=sys.stderr, flush=True)
        _warned.append(message)


def _opponent_hand(ps, p, me, bank, deck_counts, n_colors):
    """Opponent seat prefix p: (resources, dev cards, exact). Exact in 1v1 with a complete view;
    otherwise a guess that matches the public card counts."""
    missing = []

    def get(d, k):
        if k not in d:
            missing.append(k)
        return d.get(k, 0)

    res = [CARDS_PER_RESOURCE - (bank[i] if bank else 0) - get(ps, f"P{me}_{r}_IN_HAND")
           for i, r in enumerate(RESOURCES)]
    dev = {card: total - deck_counts.get(card, 0) - get(ps, f"P{me}_{card}_IN_HAND")
           - sum(ps.get(f"P{k}_PLAYED_{card}", 0) for k in range(n_colors))
           for card, total in INITIAL_DEV.items()}
    n_res, n_dev = ps.get(p + "NUM_RESOURCES_IN_HAND"), ps.get(p + "NUM_DEVELOPMENT_CARDS_IN_HAND")
    exact = (n_colors == 2 and bank is not None and not missing and min(res) >= 0 and min(dev.values()) >= 0
             and sum(res) == n_res and sum(dev.values()) == n_dev)
    if exact:
        return res, dev, True
    if n_colors == 2 and bank is not None and not missing and sum(res) != n_res:
        reason = "opponent hand reconstruction does not match the public card count"
    elif n_colors != 2:
        reason = f"{n_colors} players: hands cannot be recovered exactly"
    else:
        reason = "view is missing fields: " + ", ".join(missing[:3]) if missing else "inconsistent view"
    _warn_once(reason)
    res = _split(n_res if n_res is not None else sum(max(0, x) for x in res), res)
    weights = [dev[c] if max(dev.values()) > 0 else INITIAL_DEV[c] for c in INITIAL_DEV]
    total = n_dev if n_dev is not None else sum(max(0, x) for x in dev.values())
    dev = dict(zip(INITIAL_DEV, _split(total, weights)))
    return res, dev, False


_defaults = {}


def _default_player_state():
    """Player-state fields (without the 'P0_' prefix) and their values at the start of a game."""
    if not _defaults:
        g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
        _defaults.update({k[3:]: v for k, v in g.state.player_state.items() if k.startswith("P0_")})
    return _defaults


def complete_view(view, static_map, my_color, rng=None):
    """view: the 'state' of a decide/before message (map may be missing); static_map: the map from
    'before'. Returns a full document that catanatron.serialization.state_from_json accepts.
    The opponent's hand is exact in 1v1 (see the module docstring); if the view is inconsistent or
    incomplete, or there are more players, it is a guess matching their public card counts."""
    rng = rng or random.Random()
    doc = dict(view)
    doc["map"] = view.get("map", static_map)
    me = doc["colors"].index(my_color)
    ps = dict(doc["player_state"])
    deck_counts = dict(doc.get("development_listdeck") or {})  # composition only, e.g. {"KNIGHT": 9, ...}
    bank = doc.get("resource_freqdeck")
    for seat in range(len(doc["colors"])):
        if seat == me:
            continue
        p = f"P{seat}_"
        res, dev, _ = _opponent_hand(ps, p, me, bank, deck_counts, len(doc["colors"]))
        for r, n in zip(RESOURCES, res):
            ps[p + f"{r}_IN_HAND"] = n
        for card, n in dev.items():
            ps[p + f"{card}_IN_HAND"] = n
        for card in PLAYABLE:
            # Unknown: whether a card was bought this turn. Assume held since the turn began.
            ps[p + f"{card}_OWNED_AT_START"] = ps[p + f"{card}_IN_HAND"] > 0
        ps[p + "ACTUAL_VICTORY_POINTS"] = ps.get(p + "VICTORY_POINTS", 0) + ps[p + "VICTORY_POINT_IN_HAND"]
        ps.pop(p + "NUM_RESOURCES_IN_HAND", None)
        ps.pop(p + "NUM_DEVELOPMENT_CARDS_IN_HAND", None)
    n = len(doc["colors"])
    if bank is None:                                         # rebuild the bank from all the hands
        doc["resource_freqdeck"] = [max(0, CARDS_PER_RESOURCE - sum(ps.get(f"P{k}_{r}_IN_HAND", 0) for k in range(n)))
                                    for r in RESOURCES]
    if not view.get("development_listdeck"):                 # rebuild the deck: initial - hands - played
        deck_counts = {c: max(0, total - sum(ps.get(f"P{k}_{c}_IN_HAND", 0) + ps.get(f"P{k}_PLAYED_{c}", 0)
                                             for k in range(n)))
                       for c, total in INITIAL_DEV.items()}
    defaults = _default_player_state()
    for k in range(n):                                       # any other missing field: a new game's value
        for key, value in defaults.items():
            if f"P{k}_{key}" not in ps:
                _warn_once(f"view is missing P{k}_{key}")
                ps[f"P{k}_{key}"] = value
    doc["player_state"] = ps
    deck = [card for card, n in sorted(deck_counts.items()) for _ in range(n)]
    rng.shuffle(deck)                                          # the only real unknown: the order
    doc["development_listdeck"] = deck
    doc["game"] = dict(doc.get("game") or {})
    doc["game"].setdefault("seed", rng.randrange(2 ** 31))
    doc["random_state"] = list(random.Random(rng.randrange(2 ** 31)).getstate())
    doc["random_state"][1] = list(doc["random_state"][1])
    return doc


def rebuild_game(view, static_map, my_color, rng=None):
    """A real catanatron Game equal to the true one (except deck order and future dice)."""
    doc = complete_view(view, static_map, my_color, rng)
    players = [RandomPlayer(Color[c]) for c in doc["colors"]]   # placeholders; we decide ourselves
    return state_from_json(doc, players)
