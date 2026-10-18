"""crush2: single-file agent. crush (counter-opening + F-style value with Longest Road and port
terms) plus the improvements in catan_1v1_winning_strategy.md:

  planner      own-turn depth-2 search: trade / Year of Plenty / Monopoly -> a city or settlement it
               enables, knight -> best robber move, Road Building -> best first road
  win_check    any line this turn reaching 10 actual VP (hidden VP cards included) is taken
  knight_rule  pre-roll: play a knight only if the robber blocks us or it wins Largest Army
  robber_rule  robber target by need-weighted denial, threat-aware, protects our own tiles
  discard_rule discard surplus beyond the next build (only if the engine offers DISCARD_RESOURCE)
  trade_chain  planner: trade -> trade -> city/settlement;  road_plan: road -> settlement it opens
  dev_owned    value +3e7 per dev card ever bought (held or played); F barely values dev cards

Measured on upstream catanatron master with a local copy of the kit's clock (not the patched kit
build), paired seeds, 1000 games unless noted:
  baseline (all switches off, ~crush)            60.9%
  + planner, win_check, knight/robber rules       69.7%  (planner alone +6.1 pts, z=4.6)
  + trade_chain, road_plan                        70.3%
  + dev_owned=3e7                                 78.1%;  held-out seeds, 2000 games: 77.5% +-1.8
  mean CPU 170 ms, worst 500 ms per game (local), 0 errors

Every feature is a CONFIG switch so it can be A/B tested. Each decision falls back
full policy -> plain 1-ply value -> ROLL/END_TURN -> index 0, so nothing here can raise.
Optional data file: open_logreg.json next to this file (learned opening evaluator). Without it the
hand-written opening score is used.
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catanatron.features import build_production_features, reachability_features, resource_hand_features
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType, CITY, RESOURCES, SETTLEMENT
from catanatron.players.value import DEFAULT_WEIGHTS, value_production
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards, player_key,
                                        player_num_dev_cards, player_num_resource_cards)

CONFIG = dict(planner=True, win_check=True, knight_rule=True, robber_rule=True,
              discard_rule=False,    # untested: upstream catanatron has no DISCARD_RESOURCE. A/B on the kit
              road_plan=True,        # planner: road -> settlement on the node it opens, same turn
              trade_chain=True,      # planner: trade -> trade -> city/settlement
              reserve=0.10,          # seconds of allowance never spent on search
              cap_frac=1.0 / 6.0)     # max share of the spare allowance one decision may use
WEIGHTS = dict(lr_always=2e3, port2=3e7, port3=1e6, dev_owned=3e7)

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
COST = {"CITY": {"WHEAT": 2, "ORE": 3},
        "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
        "DEV": {"SHEEP": 1, "WHEAT": 1, "ORE": 1}}
AT = ActionType
DISCARD_RESOURCE = getattr(AT, "DISCARD_RESOURCE", None)
VP_BUILDS = (AT.BUILD_CITY, AT.BUILD_SETTLEMENT)
ENABLERS = (AT.MARITIME_TRADE, AT.PLAY_YEAR_OF_PLENTY, AT.PLAY_MONOPOLY)
WIN = 1e30
_DEBUG = bool(os.environ.get("CRUSH2_DEBUG"))   # local testing only: re-raise instead of falling back


def pips_of(number):
    return 0 if number is None else 6 - abs(7 - number)


# ---------------------------------------------------------------- value function (F + extras)
_PROD = build_production_features(True)


def make_vf(overrides=None):
    p = dict(DEFAULT_WEIGHTS)
    p.update(dict(lr_always=0.0, port2=0.0, port3=0.0, dev_bonus=0.0, army_bonus=0.0, dev_owned=0.0))
    p.update(overrides or {})

    def fn(game, color):
        st = game.state
        sample = _PROD(game, color)
        production = value_production(sample, "P0")
        enemy_production = value_production(sample, "P1", False)
        key = player_key(st, color)
        lr_len = get_longest_road_length(st, color)
        reach = reachability_features(game, color, 2)
        r0 = sum(reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
        r1 = sum(reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
        hand = resource_hand_features(game, color)
        d_city = (max(2 - hand["P0_WHEAT_IN_HAND"], 0) + max(3 - hand["P0_ORE_IN_HAND"], 0)) / 5.0
        d_sett = sum(max(1 - hand[f"P0_{r}_IN_HAND"], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4.0
        synergy = (2 - d_city - d_sett) / 2
        n_hand = player_num_resource_cards(st, color)
        owned = st.buildings_by_color[color][SETTLEMENT] + st.buildings_by_color[color][CITY]
        tiles = set()
        for n in owned:
            tiles.update(st.board.map.adjacent_tiles[n])
        n_build = len(st.board.buildable_node_ids(color))
        lr_factor = p["longest_road"] if n_build == 0 else 0.1
        v = (st.player_state[f"{key}_VICTORY_POINTS"] * p["public_vps"]
             + production * p["production"] + enemy_production * p["enemy_production"]
             + r0 * p["reachable_production_0"] + r1 * p["reachable_production_1"]
             + synergy * p["hand_synergy"] + n_build * p["buildable_nodes"] + len(tiles) * p["num_tiles"]
             + n_hand * p["hand_resources"] + (p["discard_penalty"] if n_hand > 7 else 0)
             + lr_len * lr_factor + player_num_dev_cards(st, color) * p["hand_devs"]
             + get_played_dev_cards(st, color, "KNIGHT") * p["army_size"])
        if p["lr_always"]:
            v += lr_len * p["lr_always"]
        if p["dev_bonus"] or p["army_bonus"]:
            v += (player_num_dev_cards(st, color) * p["dev_bonus"]
                  + get_played_dev_cards(st, color, "KNIGHT") * p["army_bonus"])
        if p["dev_owned"]:   # every dev card ever bought, held or played: buying is rewarded, playing is not taxed
            v += (player_num_dev_cards(st, color) + get_played_dev_cards(st, color)) * p["dev_owned"]
        if p["port2"] or p["port3"]:
            prod = {r: sample[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES}
            owned_set = set(owned)
            for res, nodes in st.board.map.port_nodes.items():
                if owned_set & nodes:
                    v += p["port3"] if res is None else p["port2"] * prod[res]
        return float(v)
    return fn


# ---------------------------------------------------------------- opening features
NAMES = ([f"pips_{r}" for r in RES] + ["pips_total", "n_res", "n_numbers"]
         + [f"port_{r}" for r in RES] + ["port_3", "port2_pips", "wood_brick", "wheat_ore",
                                          "sheep_wheat_ore", "best_open", "second_open", "n_open"])


def node_pips(board, n):
    return {r: p * 36 for r, p in board.map.node_production[n].items()}


def port_of(board):
    m = {}
    for res, nodes in board.map.port_nodes.items():
        for n in nodes:
            m[n] = res if res is not None else "3"
    return m


def open_nodes_after(board, occupied):
    """Nodes still legal for a settlement given occupied nodes (distance rule), ignoring roads."""
    blocked = set(occupied)
    for n in occupied:
        blocked.update(STATIC_GRAPH.neighbors(n))
    return [n for n in board.map.land_nodes if n not in blocked]


def side_features(board, mine, occupied, ports=None, open_nodes=None):
    ports = port_of(board) if ports is None else ports
    pips = {r: 0.0 for r in RES}
    numbers = set()
    for n in mine:
        for r, p in node_pips(board, n).items():
            pips[r] += p
        for t in board.map.adjacent_tiles[n]:
            if t.number is not None:
                numbers.add(t.number)
    f = [pips[r] for r in RES]
    f += [sum(pips.values()), sum(1 for r in RES if pips[r] > 0), len(numbers)]
    my_ports = {ports[n] for n in mine if n in ports}
    f += [1.0 if r in my_ports else 0.0 for r in RES] + [1.0 if "3" in my_ports else 0.0]
    f.append(sum(pips[r] for r in RES if r in my_ports))
    f += [min(pips["WOOD"], pips["BRICK"]), min(pips["WHEAT"], pips["ORE"]),
          min(pips["SHEEP"], pips["WHEAT"], pips["ORE"])]
    if open_nodes is None:
        open_nodes = set(open_nodes_after(board, occupied))
    near = set()
    for n in mine:
        for a in STATIC_GRAPH.neighbors(n):
            for b in STATIC_GRAPH.neighbors(a):
                if b in open_nodes:
                    near.add(b)
    vals = sorted((sum(node_pips(board, n).values()) for n in near), reverse=True)
    f += [vals[0] if vals else 0.0, vals[1] if len(vals) > 1 else 0.0, float(len(vals))]
    return f


def game_features(board, mine, theirs, i_am_first):
    occ = list(mine) + list(theirs)
    ports = port_of(board)
    open_nodes = set(open_nodes_after(board, occ))
    return (side_features(board, mine, occ, ports, open_nodes)
            + side_features(board, theirs, occ, ports, open_nodes) + [1.0 if i_am_first else 0.0])


# ---------------------------------------------------------------- F's setup reply model
def legal_nodes(board, occupied):
    return open_nodes_after(board, occupied)


def f_pick(board, theirs, occupied):
    """F's setup choice: its value is dominated by production + 4 pips per distinct resource
    (value_production with TRANSLATE_VARIETY = 4). Ties go to the lower node id."""
    have = set()
    for n in theirs:
        have.update(r for r, p in node_pips(board, n).items() if p > 0)
    best, bv = None, -math.inf
    for n in sorted(legal_nodes(board, occupied)):
        p = node_pips(board, n)
        v = sum(p.values()) + 4.0 * len(have | {r for r, x in p.items() if x > 0})
        if v > bv:
            best, bv = n, v
    return best


# ---------------------------------------------------------------- counter-opening
_W = None
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_logreg.json")
if os.path.exists(_PATH):
    try:
        with open(_PATH) as fh:
            _d = json.load(fh)
        _W = (_d["mean"], _d["std"], _d["coef"], _d["bias"])
    except Exception:
        _W = None


def open_eval(board, mine, theirs, i_first):
    x = game_features(board, mine, theirs, i_first)
    if _W is None:
        h = len(x) // 2

        def side(f):
            return f[5] + 3 * f[6] + f[17] + f[18]
        return side(x[:h]) - side(x[h:2 * h])
    mean, std, coef, bias = _W
    return bias + sum(c * (v - m) / s for c, v, m, s in zip(coef, x, mean, std))


def _rank(board, nodes, k):
    def quick(n):
        p = node_pips(board, n)
        return sum(p.values()) + 2 * len(p)
    return sorted(nodes, key=quick, reverse=True)[:k]


def choose_opening(board, cands, mine, theirs, i_first, k1=6, k2=10):
    occ = list(mine) + list(theirs)
    if not mine and not theirs:                              # first player, pick 1
        best, bv = None, -math.inf
        for c1 in _rank(board, cands, k1):
            f1 = f_pick(board, [], occ + [c1])
            f2 = f_pick(board, [f1], occ + [c1, f1])
            rest = legal_nodes(board, [c1, f1, f2])
            v = max((open_eval(board, [c1, c2], [f1, f2], True) for c2 in _rank(board, rest, k2)),
                    default=-math.inf)
            if v > bv:
                best, bv = c1, v
        return best
    if not mine and len(theirs) == 1:                        # second player, pick 1
        best, bv = None, -math.inf
        for c1 in _rank(board, cands, k1):
            v = -math.inf
            for c2 in _rank(board, legal_nodes(board, occ + [c1]), k2):
                f2 = f_pick(board, list(theirs), occ + [c1, c2])
                v = max(v, open_eval(board, [c1, c2], list(theirs) + [f2], False))
            if v > bv:
                best, bv = c1, v
        return best
    if len(mine) == 1 and len(theirs) == 1:                  # second player, pick 2
        best, bv = None, -math.inf
        for c2 in _rank(board, cands, 2 * k2):
            f2 = f_pick(board, list(theirs), occ + [c2])
            v = open_eval(board, list(mine) + [c2], list(theirs) + [f2], False)
            if v > bv:
                best, bv = c2, v
        return best
    return max(_rank(board, cands, 3 * k2),                  # first player, last pick
               key=lambda c: open_eval(board, list(mine) + [c], list(theirs), True))


def _road_spot_score(board, node, mine):
    have = {r: 0.0 for r in RES}
    for n in mine:
        for r, p in node_pips(board, n).items():
            have[r] += p
    add = node_pips(board, node)
    after = {r: have[r] + add.get(r, 0.0) for r in RES}
    s = sum(add.values()) + 3.0 * sum(1 for r in add if have[r] == 0 and add[r] > 0)
    s += min(after["WOOD"], after["BRICK"]) - min(have["WOOD"], have["BRICK"])
    s += min(after["WHEAT"], after["ORE"]) - min(have["WHEAT"], have["ORE"])
    for r, nodes in board.map.port_nodes.items():
        if node in nodes:
            s += 1.0 if r is None else 1.5 * after.get(r, 0.0) / 5.0
    return s


# ---------------------------------------------------------------- state helpers
def hand_of(st, color):
    key = player_key(st, color)
    return {r: st.player_state[f"{key}_{r}_IN_HAND"] for r in RES}


def actual_vp(st, color):
    key = player_key(st, color)
    return st.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", st.player_state[f"{key}_VICTORY_POINTS"])


def legal(g):
    """Legal actions of a game copy, whichever attribute this catanatron build keeps them in."""
    acts = getattr(g, "playable_actions", None)
    if acts is None:
        acts = getattr(g.state, "playable_actions", None)
    if acts is None:
        from catanatron.models.actions import generate_playable_actions
        acts = generate_playable_actions(g.state)
    return acts


def public_vp(st, color):
    return st.player_state[f"{player_key(st, color)}_VICTORY_POINTS"]


def knights_played(st, color):
    return get_played_dev_cards(st, color, "KNIGHT")


def dev_in_hand(st, color, card):
    return st.player_state.get(f"{player_key(st, color)}_{card}_IN_HAND", 0)


def missing(hand, cost):
    return {r: max(0, n - hand.get(r, 0)) for r, n in cost.items()}


def target_cost(st, color, hand):
    """Cost of the VP build this player is closest to (city preferred on ties)."""
    opts = []
    if st.buildings_by_color[color][SETTLEMENT]:
        opts.append((sum(missing(hand, COST["CITY"]).values()), 0, COST["CITY"]))
    opts.append((sum(missing(hand, COST["SETTLEMENT"]).values()), 1, COST["SETTLEMENT"]))
    return min(opts, key=lambda t: (t[0], t[1]))[2]


def need_weights(st, color, boost):
    hand = hand_of(st, color)
    short = missing(hand, target_cost(st, color, hand))
    return {r: (boost if short.get(r, 0) > 0 else 1.0) for r in RES}


# ---------------------------------------------------------------- agent
class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.vf = make_vf({**WEIGHTS, **CONFIG.get("vf", {})})

    def decide(self, obs, actions):
        try:
            return self._decide(obs, actions)
        except Exception:
            if _DEBUG:
                raise
        try:
            return self._one_ply(obs.sample_world(self.rng), obs.me, actions)
        except Exception:
            return _roll_or_end(actions)

    # ------------------------------------------------------------ dispatch
    def _decide(self, obs, actions):
        t0 = time.process_time()
        cpu = obs.cpu_left if obs.cpu_left is not None else 1.0
        world = obs.sample_world(self.rng)
        st = world.state
        me = obs.me
        if st.is_initial_build_phase:
            i = self._setup(st, me, actions)
            if i is not None:
                return i
        if cpu < CONFIG["reserve"]:
            return self._one_ply(world, me, actions) if cpu > 0.03 else _roll_or_end(actions)
        deadline = t0 + max(0.0, (cpu - CONFIG["reserve"]) * CONFIG["cap_frac"])
        types = {a.action_type for a in actions}
        opp = next(c for c in st.colors if c != me)

        if DISCARD_RESOURCE is not None and DISCARD_RESOURCE in types and CONFIG["discard_rule"]:
            return self._discard(st, me, actions)
        if AT.MOVE_ROBBER in types and CONFIG["robber_rule"]:
            return self._robber(st, me, opp, actions)
        if AT.ROLL in types and CONFIG["knight_rule"]:
            return self._pre_roll(st, me, opp, actions)
        if AT.END_TURN in types:
            return self._main(world, me, actions, deadline)
        return self._one_ply(world, me, actions)

    # ------------------------------------------------------------ setup
    def _setup(self, st, me, actions):
        board = st.board
        b = st.buildings_by_color
        opp = next(c for c in st.colors if c != me)
        mine = b[me][SETTLEMENT] + b[me][CITY]
        if actions[0].action_type == AT.BUILD_SETTLEMENT:
            node = choose_opening(board, [a.value for a in actions], mine, b[opp][SETTLEMENT],
                                  st.colors[0] == me)
            return next((i for i, a in enumerate(actions) if a.value == node), None)
        if actions[0].action_type == AT.BUILD_ROAD:
            open_nodes = set(board.buildable_node_ids(me, initial_build_phase=True))

            def road_score(i):
                u, v = actions[i].value
                far = v if u in mine else u
                return max((_road_spot_score(board, n, mine) for n in STATIC_GRAPH.neighbors(far)
                            if n in open_nodes), default=0.0)
            return max(range(len(actions)), key=road_score)
        return None

    # ------------------------------------------------------------ evaluation
    def _leaf(self, g, me):
        st = g.state
        if CONFIG["win_check"] and actual_vp(st, me) >= getattr(g, "vps_to_win", 10):
            return WIN
        return self.vf(g, me)

    def _one_ply(self, world, me, actions):
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                continue
            v = self._leaf(g, me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    # ------------------------------------------------------------ main phase: depth-2 planner
    def _main(self, world, me, actions, deadline):
        before = {a.action_type for a in actions}
        settle_before = {a.value for a in actions if a.action_type == AT.BUILD_SETTLEMENT}
        scored = []
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                continue
            v = self._leaf(g, me)
            if v >= WIN:
                return i
            scored.append((v, i, g))
        if not scored:
            return _roll_or_end(actions)
        best_v, best_i = max((v, -i) for v, i, _ in scored)
        best_i = -best_i
        if not CONFIG["planner"]:
            return best_i
        for v, i, g in scored:
            if time.process_time() > deadline:
                break
            t = actions[i].action_type
            if g.state.current_color() != me:
                continue
            follow = legal(g)
            if t in ENABLERS:
                cand = [b for b in follow if b.action_type in VP_BUILDS and b.action_type not in before]
                if not cand and t == AT.MARITIME_TRADE and CONFIG["trade_chain"]:
                    v3 = self._trade_chain(g, me, before, deadline)
                    if v3 > best_v:
                        best_v, best_i = v3, i
                    continue
            elif t == AT.BUILD_ROAD and CONFIG["road_plan"]:
                cand = [b for b in follow
                        if b.action_type == AT.BUILD_SETTLEMENT and b.value not in settle_before]
            elif t == AT.PLAY_KNIGHT_CARD:
                cand = [b for b in follow if b.action_type == AT.MOVE_ROBBER]
            elif t == AT.PLAY_ROAD_BUILDING:
                cand = [b for b in follow if b.action_type == AT.BUILD_ROAD]
            else:
                continue
            for b in cand:
                if time.process_time() > deadline:
                    break
                g2 = g.copy()
                try:
                    g2.execute(b)
                except Exception:
                    continue
                v2 = self._leaf(g2, me)
                if v2 > best_v:
                    best_v, best_i = v2, i
                    if v2 >= WIN:
                        return best_i
        return best_i

    def _trade_chain(self, g, me, before, deadline):
        """Best value of a second trade followed by a city/settlement it newly enables.
        Only second trades that receive a card still missing for a city or settlement."""
        st = g.state
        hand = hand_of(st, me)
        short = set()
        if st.buildings_by_color[me][SETTLEMENT] and AT.BUILD_CITY not in before:
            short.update(r for r, n in missing(hand, COST["CITY"]).items() if n)
        if AT.BUILD_SETTLEMENT not in before:
            short.update(r for r, n in missing(hand, COST["SETTLEMENT"]).items() if n)
        best = -math.inf
        for b in legal(g):
            if b.action_type != AT.MARITIME_TRADE or b.value[-1] not in short:
                continue
            if time.process_time() > deadline:
                break
            g2 = g.copy()
            try:
                g2.execute(b)
            except Exception:
                continue
            for c in legal(g2):
                if c.action_type not in VP_BUILDS or c.action_type in before:
                    continue
                g3 = g2.copy()
                try:
                    g3.execute(c)
                except Exception:
                    continue
                best = max(best, self._leaf(g3, me))
        return best

    # ------------------------------------------------------------ pre-roll knight rule
    def _pre_roll(self, st, me, opp, actions):
        roll = next(i for i, a in enumerate(actions) if a.action_type == AT.ROLL)
        knight = next((i for i, a in enumerate(actions) if a.action_type == AT.PLAY_KNIGHT_CARD), None)
        if knight is None:
            return roll
        mine_k, opp_k = knights_played(st, me), knights_played(st, opp)
        has_army = st.player_state.get(f"{player_key(st, me)}_HAS_ARMY", False)
        if not has_army and mine_k + 1 >= 3 and mine_k + 1 > opp_k:
            return knight                                     # takes Largest Army (+2 VP)
        board = st.board
        tile = board.map.land_tiles.get(board.robber_coordinate)
        blocked = 0
        if tile is not None and tile.resource is not None:
            for node in tile.nodes.values():
                bld = board.buildings.get(node)
                if bld is not None and bld[0] == me:
                    blocked += pips_of(tile.number) * (2 if bld[1] == CITY else 1)
        return knight if blocked >= 3 else roll

    # ------------------------------------------------------------ robber rule
    def _robber(self, st, me, opp, actions):
        board = st.board
        opp_hand = hand_of(st, opp)
        opp_total = sum(opp_hand.values())
        opp_max = public_vp(st, opp) + min(player_num_dev_cards(st, opp),
                                           max(0, 5 - dev_in_hand(st, me, "VICTORY_POINT")))
        threat = opp_max >= 8
        need_opp = need_weights(st, opp, 2.5 if threat else 1.5)
        need_me = need_weights(st, me, 1.5)
        steal = (sum(opp_hand[r] / opp_total * need_me[r] for r in RES) * 5.0) if opp_total else 0.0

        def score(i):
            coord, victim = actions[i].value
            tile = board.map.land_tiles[coord]
            d_opp = d_me = 0.0
            for node in tile.nodes.values():
                bld = board.buildings.get(node)
                if bld is None or tile.resource is None:    # desert: no production to deny
                    continue
                p = pips_of(tile.number)
                mult = 2 if bld[1] == CITY else 1
                if bld[0] == me:
                    d_me += p * mult * need_me[tile.resource]
                else:
                    d_opp += p * mult * need_opp[tile.resource]
            s = d_opp - d_me + (steal if victim == opp else 0.0)
            if d_me > 0 and d_opp < 2 * d_me:
                s -= 100.0
            return s
        return max(range(len(actions)), key=score)

    # ------------------------------------------------------------ discard rule
    def _discard(self, st, me, actions):
        hand = hand_of(st, me)
        cost = target_cost(st, me, hand)
        prod = {r: 0.0 for r in RES}
        for n in st.buildings_by_color[me][SETTLEMENT]:
            for r, p in node_pips(st.board, n).items():
                prod[r] += p
        for n in st.buildings_by_color[me][CITY]:
            for r, p in node_pips(st.board, n).items():
                prod[r] += 2 * p

        def score(i):
            r = actions[i].value
            if actions[i].action_type != DISCARD_RESOURCE or r not in hand:
                return (-99, -99.0)
            return (hand[r] - cost.get(r, 0), prod[r])
        return max(range(len(actions)), key=score)


def _roll_or_end(actions):
    for i, a in enumerate(actions):
        if a.action_type in (AT.ROLL, AT.END_TURN):
            return i
    return 0