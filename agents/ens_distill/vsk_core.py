"""Core of our Catan agent: fast engine copy, a static board table, a value function over hand-built
features, and a budgeted within-turn search. Variants override PARAMS."""
import math
import pickle
import random
import time
from collections import defaultdict

from catanatron.models.board import STATIC_GRAPH, Board
from catanatron.models.enums import (ActionType, ActionPrompt, ActionRecord, Action, SETTLEMENT, CITY,
                                     RESOURCES)
from catanatron.state import State

# ---------------------------------------------------------------------------------------------
# 1. Faster copies (identical semantics to the engine's pickle/deepcopy based copies).
# ---------------------------------------------------------------------------------------------


def _board_copy(self):
    b = Board(self.map, initialize=False)
    b.map = self.map
    b.buildings = self.buildings.copy()
    b.roads = self.roads.copy()
    cc = defaultdict(list)
    for c, comps in self.connected_components.items():
        cc[c] = [set(s) for s in comps]
    b.connected_components = cc
    b.board_buildable_ids = self.board_buildable_ids.copy()
    b.road_lengths = self.road_lengths.copy()
    b.road_color = self.road_color
    b.road_length = self.road_length
    b.robber_coordinate = self.robber_coordinate
    b.buildable_subgraph = self.buildable_subgraph
    b.buildable_edges_cache = {k: list(v) for k, v in self.buildable_edges_cache.items()}
    b.player_port_resources_cache = {k: set(v) for k, v in self.player_port_resources_cache.items()}
    return b


_orig_state_copy = State.copy


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


Board.copy = _board_copy
State.copy = _state_copy

# ---------------------------------------------------------------------------------------------
# 2. Static tables.
# ---------------------------------------------------------------------------------------------
RIDX = {r: i for i, r in enumerate(RESOURCES)}   # WOOD BRICK SHEEP WHEAT ORE
NEIGH = {n: tuple(STATIC_GRAPH.neighbors(n)) for n in STATIC_GRAPH.nodes}
PROBA = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
DEV_DECK = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}
DEVS = list(DEV_DECK)


class MapInfo:
    def __init__(self, cmap):
        self.node_tiles = {}          # node -> list of (coord, ridx, prob)
        self.node_prod = {}           # node -> [p_r]*5 ignoring robber
        self.node_total = {}
        tile_coord = {}
        for coord, tile in cmap.land_tiles.items():
            tile_coord[id(tile)] = coord
        for n in cmap.land_nodes:
            lst = []
            prod = [0.0] * 5
            for t in cmap.adjacent_tiles[n]:
                if t.resource is None:
                    continue
                p = PROBA[t.number] / 36.0
                r = RIDX[t.resource]
                lst.append((tile_coord[id(t)], r, p))
                prod[r] += p
            self.node_tiles[n] = lst
            self.node_prod[n] = prod
            self.node_total[n] = sum(prod)
        self.port_of = {}             # node -> ridx, or 5 for 3:1
        for res, nodes in cmap.port_nodes.items():
            for n in nodes:
                self.port_of[n] = 5 if res is None else RIDX[res]
        self.tile_nodes = {coord: tuple(t.nodes.values()) for coord, t in cmap.land_tiles.items()}
        self.tile_res = {coord: (None if t.resource is None else RIDX[t.resource]) for coord, t in cmap.land_tiles.items()}
        self.tile_prob = {coord: (0 if t.number is None else PROBA[t.number] / 36.0) for coord, t in cmap.land_tiles.items()}
        # board-wide scarcity: total pips per resource
        tot = [0.0] * 5
        for coord, t in cmap.land_tiles.items():
            if t.resource is not None:
                tot[RIDX[t.resource]] += PROBA[t.number] / 36.0
        self.res_total = tot
        self.land_nodes = cmap.land_nodes


_MAP_CACHE = {}


def map_info(cmap):
    key = id(cmap)
    mi = _MAP_CACHE.get(key)
    if mi is not None and mi[0] is cmap:
        return mi[1]
    sig = tuple(sorted((c, t.resource, t.number) for c, t in cmap.land_tiles.items()))
    for k, (m, info, s) in list(_MAP_CACHE.items()):
        if s == sig:
            _MAP_CACHE[key] = (cmap, info, sig)
            return info
    info = MapInfo(cmap)
    if len(_MAP_CACHE) > 64:
        _MAP_CACHE.clear()
    _MAP_CACHE[key] = (cmap, info, sig)
    return info


# ---------------------------------------------------------------------------------------------
# 3. Features and value.
# ---------------------------------------------------------------------------------------------
F_WEIGHTS = dict(
    public_vps=3e14, production=1e8, enemy_production=-1e8, num_tiles=1.0,
    reachable_production_1=1e4, buildable_nodes=1e3, longest_road=10.0,
    hand_synergy=1e2, hand_resources=1.0, discard_penalty=-5.0, hand_devs=10.0, army_size=10.1,
    # extensions (0 = pure F)
    hidden_vp=0.0,          # my VP cards counted at this weight (F: ignored except hand_devs)
    knight_hand=0.0,        # extra per unplayed knight
    devcard=0.0,            # extra per unplayed non-VP dev card
    la_progress=0.0,        # value of being on track for largest army
    lr_progress=0.0,        # value of road length lead when contested
    enemy_reach=0.0,        # penalty on enemy reachable production
    port_synergy=0.0,       # production through a matching 2:1 port / 3:1 port
    enemy_vps=0.0,          # weight on opponent public VPs (negative)
    city_spots=0.0,         # settlements that can still be upgraded weighted by production
    prod_res=(1.0, 1.0, 1.0, 1.0, 1.0),
    discard_per_card=0.0,   # penalty per card at risk when holding > 7
    minres=0.0,             # weight on my weakest resource production (balance)
    city_guard=0.0,
    lr_defense=0.0,         # keep a margin once I hold longest road         # bonus for keeping a near-complete city hand (<=1 card missing)
    variety_w=4 * 0.02778,  # production-units per distinct resource (F: 4 pips)
    discard_turn_only=0.0,
)


class Ctx:
    """Per-decision context: map info and perspective."""
    __slots__ = ("mi", "me", "opp", "W", "opp_hidden_vp_exp")


def _reach1(state, color, mi, board, ps_key):
    comps = board.connected_components[color]
    zero = set()
    for c in comps:
        zero |= c
    buildings = board.buildings
    roads = board.roads
    level = set(zero)
    for n in zero:
        bd = buildings.get(n)
        if bd is not None and bd[0] != color:
            continue
        for m in NEIGH[n]:
            rc = roads.get((n, m))
            if rc is None or rc == color:
                level.add(m)
    bb = board.board_buildable_ids
    tot = 0.0
    np_ = mi.node_prod
    for n in level:
        if n in bb:
            tot += mi.node_total[n]
        else:
            bd = buildings.get(n)
            if bd is not None and bd[0] == color:
                tot += mi.node_total[n]
    return tot


def evaluate(game, ctx):
    W = ctx.W
    st = game.state
    ps = st.player_state
    board = st.board
    mi = ctx.mi
    me, opp = ctx.me, ctx.opp
    ki = "P%d_" % st.color_to_index[me]
    ko = "P%d_" % st.color_to_index[opp]
    robber = board.robber_coordinate

    # production (effective) for both
    def prod(color):
        bbc = st.buildings_by_color[color]
        pr = [0.0, 0.0, 0.0, 0.0, 0.0]
        nt = mi.node_tiles
        tiles = set()
        for n in bbc[SETTLEMENT]:
            for (c, r, p) in nt[n]:
                tiles.add(c)
                if c != robber:
                    pr[r] += p
        for n in bbc[CITY]:
            for (c, r, p) in nt[n]:
                tiles.add(c)
                if c != robber:
                    pr[r] += 2 * p
        return pr, tiles

    pr_me, tiles_me = prod(me)
    pr_op, _ = prod(opp)
    variety = sum(1 for x in pr_me if x != 0)
    prw = W["prod_res"]
    production = sum(pr_me[i] * prw[i] for i in range(5)) + variety * W["variety_w"] + W["minres"] * min(pr_me)
    eprw = W.get("eprw")
    enemy_production = sum(pr_op) if not eprw else sum(pr_op[i] * eprw[i] for i in range(5))

    wood = ps[ki + "WOOD_IN_HAND"]; brick = ps[ki + "BRICK_IN_HAND"]; sheep = ps[ki + "SHEEP_IN_HAND"]
    wheat = ps[ki + "WHEAT_IN_HAND"]; ore = ps[ki + "ORE_IN_HAND"]
    dist_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
    dist_sett = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
    synergy = (2 - dist_city - dist_sett) / 2
    nhand = wood + brick + sheep + wheat + ore

    buildable = board.buildable_node_ids(me)
    nbuild = len(buildable)
    reach1 = _reach1(st, me, mi, board, ki)
    lr = ps[ki + "LONGEST_ROAD_LENGTH"]
    lr_factor = W["longest_road"] if nbuild == 0 else 0.1
    ndev = (ps[ki + "YEAR_OF_PLENTY_IN_HAND"] + ps[ki + "MONOPOLY_IN_HAND"] + ps[ki + "VICTORY_POINT_IN_HAND"]
            + ps[ki + "KNIGHT_IN_HAND"] + ps[ki + "ROAD_BUILDING_IN_HAND"])
    knights = ps[ki + "PLAYED_KNIGHT"]

    v = (ps[ki + "VICTORY_POINTS"] * W["public_vps"]
         + production * W["production"]
         + enemy_production * W["enemy_production"]
         + reach1 * W["reachable_production_1"]
         + synergy * W["hand_synergy"]
         + nbuild * W["buildable_nodes"]
         + len(tiles_me) * W["num_tiles"]
         + nhand * W["hand_resources"]
         + (W["discard_penalty"] if nhand > 7 else 0.0)
         + lr * lr_factor
         + ndev * W["hand_devs"]
         + knights * W["army_size"])

    # ---- extensions ----
    if W["hidden_vp"]:
        v += ps[ki + "VICTORY_POINT_IN_HAND"] * W["hidden_vp"]
    if W["city_guard"] and ps[ki + "CITIES_AVAILABLE"] > 0 and st.buildings_by_color[me][SETTLEMENT]:
        miss = max(2 - wheat, 0) + max(3 - ore, 0)
        if miss <= 1:
            v += W["city_guard"] * (2 - miss) / 2.0
    if W["discard_per_card"] and nhand > 7:
        v -= W["discard_per_card"] * (nhand // 2)
    if W["knight_hand"]:
        v += ps[ki + "KNIGHT_IN_HAND"] * W["knight_hand"]
    if W["devcard"]:
        v += (ps[ki + "YEAR_OF_PLENTY_IN_HAND"] + ps[ki + "MONOPOLY_IN_HAND"] + ps[ki + "ROAD_BUILDING_IN_HAND"]) * W["devcard"]
    if W["la_progress"]:
        ok = ps[ko + "PLAYED_KNIGHT"]
        mine = knights + ps[ki + "KNIGHT_IN_HAND"]
        if not ps[ki + "HAS_ARMY"]:
            need = max(3, ok + 1)
            v += W["la_progress"] * min(mine, need) / need
    if W["lr_progress"]:
        olr = ps[ko + "LONGEST_ROAD_LENGTH"]
        if not ps[ki + "HAS_ROAD"]:
            need = max(5, olr + 1)
            v += W["lr_progress"] * min(lr, need) / need
        elif W.get("lr_defense"):
            v += W["lr_defense"] * min(max(lr - olr, 0), 2) / 2.0
    if W["enemy_reach"]:
        v += W["enemy_reach"] * _reach1(st, opp, mi, board, ko)
    if W["enemy_vps"]:
        v += W["enemy_vps"] * ps[ko + "VICTORY_POINTS"]
    if W["port_synergy"]:
        ports = board.get_player_port_resources(me)
        if ports:
            best = 0.0
            for res in ports:
                if res is None:
                    best = max(best, sum(pr_me) / 3.0 * 0.5)
                else:
                    best = max(best, pr_me[RIDX[res]])
            v += W["port_synergy"] * best
    return v


def is_my_turn_free(game, me):
    st = game.state
    return st.current_color() == me and st.current_prompt == ActionPrompt.PLAY_TURN and not st.is_road_building


# ---------------------------------------------------------------------------------------------
# 4. Search.
# ---------------------------------------------------------------------------------------------
AT = ActionType
DEFAULT_SEARCH = dict(
    depth=2,            # extra plies of my own in-turn actions beyond the root
    beam=3,             # children expanded per node
    frac=0.10,          # share of the remaining clock one decision may spend
    cap=0.060,          # hard cap per decision (s)
    floor=0.0015,
    reserve=0.040,
    steal_bonus=10.0,
    dev_expect=True,    # value a dev purchase by its expectation over the unseen pile
    resolve_forced=True,
)


class SearchAgent:
    W = dict(F_WEIGHTS)
    S = dict(DEFAULT_SEARCH)
    PLACER = None

    def __init__(self):
        self.rng = random.Random(0)
        self._warm()

    # warm-up so first-game decisions are not charged for imports and caches
    def _warm(self):
        try:
            from catanatron.game import Game
            from catanatron.models.player import Color, RandomPlayer
            g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
            st = random.getstate()
            for _ in range(60):
                if g.winning_color() is not None:
                    break
                acts = g.playable_actions
                ctx = self._ctx(g, g.state.current_color())
                for a in acts[:5]:
                    c = g.copy()
                    try:
                        c.execute(a)
                        self.value(c, ctx)
                    except Exception:
                        pass
                g.execute(acts[0])
            random.setstate(st)
        except Exception:
            pass

    def value(self, game, ctx):
        return evaluate(game, ctx)

    def _ctx(self, game, me):
        ctx = Ctx()
        ctx.mi = map_info(game.state.board.map)
        ctx.me = me
        ctx.opp = [c for c in game.state.colors if c != me][0]
        ctx.W = self.W
        ctx.opp_hidden_vp_exp = 0.0
        return ctx

    # ---- action application with chance handling ----
    def _unseen_devs(self, game, me):
        st = game.state
        ps = st.player_state
        ki = "P%d_" % st.color_to_index[me]
        ko = "P%d_" % (1 - st.color_to_index[me])
        cnt = {}
        for d in DEVS:
            c = DEV_DECK[d] - ps[ki + d + "_IN_HAND"] - ps.get(ki + "PLAYED_" + d, 0) - ps.get(ko + "PLAYED_" + d, 0)
            if c > 0:
                cnt[d] = c
        return cnt

    def _value_after(self, game, action, ctx, deadline):
        """Returns (value, child_state_or_None). child is a state from which my turn can continue."""
        t = action.action_type
        S = self.S
        if t == AT.ROLL:
            return self.value(game, ctx), None
        if t == AT.END_TURN:
            return self.value(game, ctx), None
        if t == AT.BUY_DEVELOPMENT_CARD and S["dev_expect"]:
            cnt = self._unseen_devs(game, ctx.me)
            tot = sum(cnt.values())
            if tot == 0:
                c = game.copy(); c.execute(action)
                return self.value(c, ctx), c
            ev = 0.0
            child = None
            best_c = -1
            for d, c_ in cnt.items():
                g2 = game.copy()
                g2.execute(action, validate_action=False, action_record=ActionRecord(action, d))
                ev += c_ / tot * self.value(g2, ctx)
                if c_ > best_c:
                    best_c, child = c_, g2
            return ev, child
        if t == AT.MOVE_ROBBER:
            coord, victim = action.value
            g2 = game.copy()
            if victim is not None and S.get("steal_repr"):
                ps = game.state.player_state
                kv = "P%d_" % game.state.color_to_index[victim]
                res = max(RESOURCES, key=lambda r: ps[kv + r + "_IN_HAND"])
                g2.execute(action, validate_action=False, action_record=ActionRecord(action, res))
                return self.value(g2, ctx), g2
            g2.execute(Action(action.color, AT.MOVE_ROBBER, (coord, None)), validate_action=False)
            v = self.value(g2, ctx)
            if victim is not None:
                v += S["steal_bonus"]
            return v, g2
        g2 = game.copy()
        g2.execute(action)
        if S["resolve_forced"]:
            self._resolve(g2, ctx, deadline)
        return self.value(g2, ctx), g2

    def _resolve(self, g, ctx, deadline, limit=4):
        """After a knight / road-building card the same player must choose again (robber, free
        roads). Resolve those greedily so the card is valued by its best use."""
        me = ctx.me
        for _ in range(limit):
            st = g.state
            if g.winning_color() is not None or st.current_color() != me:
                return
            if not (st.current_prompt == ActionPrompt.MOVE_ROBBER or st.is_road_building):
                return
            best, bv = None, None
            for a in g.playable_actions:
                if a.action_type == AT.MOVE_ROBBER:
                    v, c = self._value_after(g, a, ctx, deadline)
                else:
                    v, c = self._plain(g, a, ctx)
                if bv is None or v > bv:
                    best, bv = c, v
            if best is None:
                return
            g.state = best.state
            g.playable_actions = best.playable_actions

    def _plain(self, game, action, ctx):
        g2 = game.copy()
        g2.execute(action)
        return self.value(g2, ctx), g2

    def _expand(self, game, ctx, depth, deadline):
        """Best value reachable from `game` continuing my turn (or stopping now)."""
        base = self.value(game, ctx)
        if depth <= 0 or time.process_time() > deadline:
            return base
        acts = [a for a in game.playable_actions if a.action_type not in (AT.END_TURN, AT.ROLL)]
        if not acts:
            return base
        scored = []
        for a in acts:
            if time.process_time() > deadline:
                break
            try:
                v, child = self._value_after(game, a, ctx, deadline)
            except Exception:
                continue
            if child is not None and child.winning_color() == ctx.me:
                v = max(v, 1e18)
            scored.append((v, child))
        if not scored:
            return base
        scored.sort(key=lambda x: -x[0])
        best = max(base, scored[0][0])
        if depth > 1:
            for v, child in scored[: self.S["beam"]]:
                if child is None or time.process_time() > deadline:
                    continue
                if child.winning_color() is not None:
                    continue
                if is_my_turn_free(child, ctx.me):
                    best = max(best, self._expand(child, ctx, depth - 1, deadline))
        return best

    def _combo(self, g, ctx, depth, deadline):
        """Best value of completing a VP build (settlement/city) from g, allowing up to depth-1 more
        bank trades first. None if no build is reachable."""
        best = None
        acts = g.playable_actions
        for a in acts:
            t = a.action_type
            if t == AT.BUILD_SETTLEMENT or t == AT.BUILD_CITY:
                v, _ = self._plain(g, a, ctx)
                if best is None or v > best:
                    best = v
        if best is not None or depth <= 1:
            return best
        for a in acts:
            if a.action_type != AT.MARITIME_TRADE or time.process_time() > deadline:
                continue
            _, c = self._plain(g, a, ctx)
            v = self._combo(c, ctx, depth - 1, deadline)
            if v is not None and (best is None or v > best):
                best = v
        return best

    def budget(self, obs, n_actions, game):
        S = self.S
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        b = (left - S["reserve"]) * S["frac"]
        if game.state.is_initial_build_phase:
            b = (left - S["reserve"]) * 0.25
        return max(S["floor"], min(S["cap"] * (2.5 if game.state.is_initial_build_phase else 1.0), b))

    def decide(self, obs, playable_actions):
        t0 = time.process_time()
        try:
            return self._decide(obs, playable_actions, t0)
        except Exception:
            return 0

    def _decide(self, obs, actions, t0):
        game = obs._game
        me = obs.me
        ctx = self._ctx(game, me)
        budget = self.budget(obs, len(actions), game)
        deadline = t0 + budget
        st = game.state
        if st.is_initial_build_phase and self.PLACER is not None:
            try:
                i = opening_decision(self.PLACER, game, actions, me, self.S.get("n_cands", 25))
                if i is not None:
                    return i
            except Exception:
                pass
        # root: 1-ply values for all actions
        scored = []
        for i, a in enumerate(actions):
            try:
                v, child = self._value_after(game, a, ctx, deadline + budget)
            except Exception:
                continue
            if child is not None and child.winning_color() == me:
                return i
            scored.append([v, i, child])
        if not scored:
            return 0
        combo = self.S.get("combo", 0)
        if combo and is_my_turn_free(game, me):
            for entry in scored:
                a = actions[entry[1]]
                if entry[2] is None or time.process_time() > deadline:
                    continue
                if a.action_type in (AT.MARITIME_TRADE, AT.PLAY_YEAR_OF_PLENTY, AT.PLAY_MONOPOLY, AT.BUILD_ROAD):
                    if not is_my_turn_free(entry[2], me):
                        continue
                    cv = self._combo(entry[2], ctx, combo if a.action_type != AT.BUILD_ROAD else 1, deadline)
                    if cv is not None and cv > entry[0]:
                        entry[0] = cv
                        entry.append(True)
                    elif a.action_type == AT.MARITIME_TRADE and self.S.get("trade_only_combo"):
                        ps = game.state.player_state
                        k = "P%d_" % game.state.color_to_index[me]
                        nh = sum(ps[k + r + "_IN_HAND"] for r in RESOURCES)
                        if nh <= self.S.get("trade_hold", 7):
                            entry[0] = -1e30
        rs = self.S.get("road_save", 0.0)
        if rs and is_my_turn_free(game, me):
            ps = game.state.player_state
            k = "P%d_" % game.state.color_to_index[me]
            if ps[k + "SETTLEMENTS_AVAILABLE"] > 0 and game.state.board.buildable_node_ids(me):
                for entry in scored:
                    if actions[entry[1]].action_type == AT.BUILD_ROAD and len(entry) == 3:
                        entry[0] -= rs
        ds, dc = self.S.get("dev_save_sett", 0.0), self.S.get("dev_save_city", 0.0)
        if (ds or dc) and is_my_turn_free(game, me):
            st = game.state
            ps = st.player_state
            k = "P%d_" % st.color_to_index[me]
            wheat, ore, sheep = ps[k + "WHEAT_IN_HAND"], ps[k + "ORE_IN_HAND"], ps[k + "SHEEP_IN_HAND"]
            pen = 0.0
            if ds and ps[k + "SETTLEMENTS_AVAILABLE"] > 0 and (sheep == 1 or wheat == 1) \
                    and ps[k + "WOOD_IN_HAND"] >= 1 and ps[k + "BRICK_IN_HAND"] >= 1 and st.board.buildable_node_ids(me):
                pen = ds
            if dc and ps[k + "CITIES_AVAILABLE"] > 0 and st.buildings_by_color[me][SETTLEMENT] \
                    and max(2 - wheat, 0) + max(3 - ore, 0) <= 1:
                pen = max(pen, dc)
            if pen:
                for entry in scored:
                    if actions[entry[1]].action_type == AT.BUY_DEVELOPMENT_CARD:
                        entry[0] -= pen
        if is_my_turn_free(game, me) and self.S["depth"] > 0:
            order = sorted(scored, key=lambda x: -x[0])
            for entry in order[: self.S["beam"] + 1]:
                if time.process_time() > deadline:
                    break
                child = entry[2]
                if child is None or child.winning_color() is not None:
                    continue
                if is_my_turn_free(child, me):
                    entry[0] = max(entry[0], self._expand(child, ctx, self.S["depth"], deadline))
        best = max(scored, key=lambda x: (x[0], -x[1]))
        return best[1]


def make_agent(weights=None, search=None, place_model=None):
    W = dict(F_WEIGHTS); W.update(weights or {})
    S = dict(DEFAULT_SEARCH); S.update(search or {})

    class Agent(SearchAgent):
        pass
    Agent.W = W
    Agent.S = S
    if place_model:
        import os as _o, pickle as _p
        Agent.PLACER = Placer(_p.load(open(place_model, "rb")))
    return Agent


# ---------------------------------------------------------------------------------------------
# 5. Learned-value features (both perspectives; no hidden information used for the opponent).
# ---------------------------------------------------------------------------------------------
UNSEEN_VP_FRAC_DEFAULT = 5.0 / 25.0


def _expand_level(board, color, start, mi):
    """Nodes reachable with one more road from `start` (not through enemy nodes/roads)."""
    buildings = board.buildings
    roads = board.roads
    out = set()
    for n in start:
        bd = buildings.get(n)
        if bd is not None and bd[0] != color:
            continue
        for m in NEIGH[n]:
            rc = roads.get((n, m))
            if rc is None or rc == color:
                out.add(m)
    return out


def player_features(st, color, other, mi, perspective_me, hidden_vp_exp=0.0):
    ps = st.player_state
    board = st.board
    k = "P%d_" % st.color_to_index[color]
    robber = board.robber_coordinate
    bbc = st.buildings_by_color[color]
    setts = bbc[SETTLEMENT]
    cities = bbc[CITY]
    nt = mi.node_tiles
    pr = [0.0] * 5
    pr_nr = 0.0
    for n in setts:
        for (c, r, p) in nt[n]:
            pr_nr += p
            if c != robber:
                pr[r] += p
    for n in cities:
        for (c, r, p) in nt[n]:
            pr_nr += 2 * p
            if c != robber:
                pr[r] += 2 * p
    ptot = sum(pr)
    variety = sum(1 for x in pr if x > 0)
    minres = min(pr)
    hand = [ps[k + "WOOD_IN_HAND"], ps[k + "BRICK_IN_HAND"], ps[k + "SHEEP_IN_HAND"], ps[k + "WHEAT_IN_HAND"], ps[k + "ORE_IN_HAND"]]
    nh = sum(hand)
    wood, brick, sheep, wheat, ore = hand
    d_city = max(2 - wheat, 0) + max(3 - ore, 0)
    d_sett = max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)
    d_dev = max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - ore, 0)
    if perspective_me:
        vp = ps[k + "ACTUAL_VICTORY_POINTS"]
        vpcards = ps[k + "VICTORY_POINT_IN_HAND"]
    else:
        vp = ps[k + "VICTORY_POINTS"] + hidden_vp_exp
        vpcards = hidden_vp_exp
    knights_h = ps[k + "KNIGHT_IN_HAND"] if perspective_me else 0.0
    ndev = (ps[k + "YEAR_OF_PLENTY_IN_HAND"] + ps[k + "MONOPOLY_IN_HAND"] + ps[k + "VICTORY_POINT_IN_HAND"]
            + ps[k + "KNIGHT_IN_HAND"] + ps[k + "ROAD_BUILDING_IN_HAND"])
    other_dev = ndev - (ps[k + "VICTORY_POINT_IN_HAND"] + ps[k + "KNIGHT_IN_HAND"]) if perspective_me else 0.0
    kp = ps[k + "PLAYED_KNIGHT"]
    has_la = 1.0 if ps[k + "HAS_ARMY"] else 0.0
    has_lr = 1.0 if ps[k + "HAS_ROAD"] else 0.0
    lr = ps[k + "LONGEST_ROAD_LENGTH"]
    # expansion
    comps = board.connected_components[color]
    zero = set()
    for c in comps:
        zero |= c
    bb = board.board_buildable_ids
    lvl0 = [n for n in zero if n in bb]
    best0 = max((mi.node_total[n] for n in lvl0), default=0.0)
    l1 = _expand_level(board, color, zero, mi)
    lvl1 = [n for n in l1 if n in bb and n not in zero]
    best1 = max((mi.node_total[n] for n in lvl1), default=0.0)
    l2 = _expand_level(board, color, l1, mi)
    lvl2 = [n for n in l2 if n in bb and n not in l1 and n not in zero]
    best2 = max((mi.node_total[n] for n in lvl2), default=0.0)
    sett_av = ps[k + "SETTLEMENTS_AVAILABLE"]
    city_av = ps[k + "CITIES_AVAILABLE"]
    roads_av = ps[k + "ROADS_AVAILABLE"]
    # ports
    port3 = 0.0
    port2 = 0.0
    for n in setts + cities:
        po = mi.port_of.get(n)
        if po is None:
            continue
        if po == 5:
            port3 = 1.0
        else:
            port2 += pr[po] if pr[po] > 0 else 0.02
    robbed = 1.0 if pr_nr - ptot > 1e-9 else 0.0
    # best city upgrade available (production of best settlement)
    best_up = 0.0
    for n in setts:
        s = 0.0
        for (c, r, p) in nt[n]:
            if c != robber:
                s += p
        best_up = max(best_up, s)
    return [
        vp, vpcards, ptot, pr[0], pr[1], pr[2], pr[3], pr[4], variety, minres, pr_nr,
        nh, wood, brick, sheep, wheat, ore, 1.0 if nh > 7 else 0.0,
        d_city, d_sett, d_dev, 1.0 if d_city == 0 else 0.0, 1.0 if d_sett == 0 else 0.0,
        knights_h, other_dev, ndev, kp, has_la, has_lr, lr,
        len(lvl0), best0, len(lvl1), best1, best2, sett_av, city_av, roads_av,
        port3, port2, robbed, best_up, len(setts), len(cities),
    ]


PF_NAMES = ["vp", "vpcards", "ptot", "pwood", "pbrick", "psheep", "pwheat", "pore", "variety", "minres", "pr_nr",
            "nh", "hwood", "hbrick", "hsheep", "hwheat", "hore", "over7",
            "d_city", "d_sett", "d_dev", "can_city", "can_sett",
            "knights_h", "other_dev", "ndev", "kp", "has_la", "has_lr", "lr",
            "n_lvl0", "best0", "n_lvl1", "best1", "best2", "sett_av", "city_av", "roads_av",
            "port3", "port2", "robbed", "best_up", "nsett", "ncity"]


def hidden_vp_expectation(st, me):
    """Expected number of VP cards among the opponent's unplayed devs, from public info."""
    ps = st.player_state
    i = st.color_to_index[me]
    ki, ko = "P%d_" % i, "P%d_" % (1 - i)
    n_opp = sum(ps[ko + d + "_IN_HAND"] for d in DEVS)
    if n_opp == 0:
        return 0.0
    vp_unseen = 5 - ps[ki + "VICTORY_POINT_IN_HAND"]
    unseen = 25
    for d in DEVS:
        unseen -= ps[ki + d + "_IN_HAND"] + ps.get(ki + "PLAYED_" + d, 0) + ps.get(ko + "PLAYED_" + d, 0)
    if unseen <= 0:
        return 0.0
    return n_opp * vp_unseen / unseen


def state_features(game, me, mi):
    st = game.state
    opp = st.colors[1 - st.color_to_index[me]]
    hv = hidden_vp_expectation(st, me)
    fm = player_features(st, me, opp, mi, True)
    fo = player_features(st, opp, me, mi, False, hv)
    my_turn = 1.0 if st.colors[st.current_turn_index] == me else 0.0
    vd = fm[0] - fo[0]
    shared = [my_turn, vd, max(fm[0], fo[0]), fm[2] - fo[2], 1.0 if fm[0] >= 8 else 0.0, 1.0 if fo[0] >= 8 else 0.0,
              st.num_turns / 100.0, sum(st.resource_freqdeck) / 95.0, len(st.development_listdeck) / 25.0]
    return fm + fo + shared


SHARED_NAMES = ["my_turn", "vdiff", "vmax", "pdiff", "me8", "op8", "turn", "bank", "deck"]
FEATURE_NAMES = ["m_" + n for n in PF_NAMES] + ["o_" + n for n in PF_NAMES] + SHARED_NAMES


# ---------------------------------------------------------------------------------------------
# 6. Learned value agent.
# ---------------------------------------------------------------------------------------------
import os as _os
import numpy as _np


class LearnedAgent(SearchAgent):
    MODEL = None       # dict loaded from pickle
    KIND = "lr"

    def __init__(self):
        m = self.MODEL[self.KIND]
        if self.KIND == "lr":
            self._w = list(m["w"]); self._b = m["b"]
        else:
            self._mu = _np.array(m["mu"]); self._sd = _np.array(m["sd"])
            self._W1 = _np.array(m["W1"]) / self._sd[:, None]
            self._b1 = _np.array(m["b1"]) - (self._mu / self._sd) @ _np.array(m["W1"])
            self._W2 = _np.array(m["W2"]); self._b2 = m["b2"]
        super().__init__()

    def value(self, game, ctx):
        w = game.winning_color()
        if w is not None:
            return 1e9 if w == ctx.me else -1e9
        f = state_features(game, ctx.me, ctx.mi)
        if self.KIND == "lr":
            s = self._b
            for a, b in zip(self._w, f):
                s += a * b
            return s
        h = _np.asarray(f) @ self._W1 + self._b1
        _np.maximum(h, 0, out=h)
        return float(h @ self._W2 + self._b2)


def make_learned(model_path, kind="lr", search=None):
    import pickle as _pk
    S = dict(DEFAULT_SEARCH); S.update(search or {})
    model = _pk.load(open(model_path, "rb"))

    class Agent(LearnedAgent):
        pass
    Agent.MODEL = model
    Agent.KIND = kind
    Agent.S = S
    return Agent


# ---------------------------------------------------------------------------------------------
# 7. Opening placement planner: a placement-outcome model + a simulated draft.
# ---------------------------------------------------------------------------------------------
class Placer:
    def __init__(self, model):
        self.sel = model["sel"]
        n = len(self.sel)
        w = model["w"]
        self.wm = dict(zip(self.sel, w[:n]))
        self.wo = dict(zip(self.sel, w[n:2 * n]))
        self.wfirst = w[2 * n] if len(w) > 2 * n else 0.0

    # --- geometry helpers on a lightweight occupancy model ---
    @staticmethod
    def buildable(n, occ):
        if n in occ:
            return False
        for m in NEIGH[n]:
            if m in occ:
                return False
        return True

    def best_dir(self, mi, s, occ, roads_taken, fixed_road=None):
        """(best1, best2) for a settlement at s with its initial road chosen (or fixed)."""
        land = mi.land_nodes
        dirs = [fixed_road] if fixed_road is not None else [m for m in NEIGH[s] if m in land and (s, m) not in roads_taken]
        best = (0.0, 0.0)
        for nb in dirs:
            if nb is None:
                continue
            b1 = 0.0
            lvl1 = []
            for x in NEIGH[nb]:
                if x == s or x not in land:
                    continue
                lvl1.append(x)
                if self.buildable(x, occ):
                    b1 = max(b1, mi.node_total[x])
            b2 = 0.0
            for x in lvl1:
                for y in NEIGH[x]:
                    if y in (s, nb) or y not in land or y in lvl1:
                        continue
                    if self.buildable(y, occ):
                        b2 = max(b2, mi.node_total[y])
            if (b1, b2) > best:
                best = (b1, b2)
        return best

    def pfeats(self, mi, nodes, occ, roads=None, second=None):
        """Feature dict for a player owning settlements `nodes` (roads: {node: neighbour} for
        settlements whose road is already placed)."""
        pr = [0.0] * 5
        for n in nodes:
            p = mi.node_prod[n]
            for r in range(5):
                pr[r] += p[r]
        f = {}
        f["ptot"] = sum(pr)
        f["pwood"], f["pbrick"], f["psheep"], f["pwheat"], f["pore"] = pr
        f["variety"] = sum(1 for x in pr if x > 0)
        f["minres"] = min(pr)
        port3, port2 = 0.0, 0.0
        for n in nodes:
            po = mi.port_of.get(n)
            if po is None:
                continue
            if po == 5:
                port3 = 1.0
            else:
                port2 += pr[po] if pr[po] > 0 else 0.02
        f["port3"], f["port2"] = port3, port2
        b1, b2 = 0.0, 0.0
        for n in nodes:
            fr = (roads or {}).get(n)
            x1, x2 = self.best_dir(mi, n, occ, set(), fr)
            b1, b2 = max(b1, x1), max(b2, x2)
        f["best1"], f["best2"] = b1, b2
        f["n_lvl1"] = 0.0
        f["n_lvl0"] = 0.0
        f["best0"] = 0.0
        hand = [0, 0, 0, 0, 0]
        if second is not None:
            for (c, r, p) in mi.node_tiles[second]:
                hand[r] += 1
        f["hwood"], f["hbrick"], f["hsheep"], f["hwheat"], f["hore"] = hand
        return f

    def score(self, mi, mine, opp, occ, my_roads=None, opp_roads=None, my_second=None, opp_second=None):
        fm = self.pfeats(mi, mine, occ, my_roads, my_second)
        fo = self.pfeats(mi, opp, occ, opp_roads, opp_second)
        s = 0.0
        for k in self.sel:
            s += self.wm[k] * fm[k] + self.wo[k] * fo[k]
        return s

    @staticmethod
    def f_pick(mi, opp_nodes, occ, candidates):
        """F's greedy opening choice: production + 0.111 * variety of its resulting set."""
        best, bv = None, None
        base = [0.0] * 5
        for n in opp_nodes:
            for r in range(5):
                base[r] += mi.node_prod[n][r]
        for n in candidates:
            if not Placer.buildable(n, occ):
                continue
            p = mi.node_prod[n]
            tot = 0.0
            var = 0
            for r in range(5):
                x = base[r] + p[r]
                tot += x
                if x > 0:
                    var += 1
            v = tot + var * 4 * 0.02778
            if bv is None or v > bv + 1e-12:
                best, bv = n, v
        return best

    def my_best_second(self, mi, first, opp, occ, opp_roads=None, my_roads=None):
        best, bv = None, None
        for n in mi.land_nodes:
            if not self.buildable(n, occ):
                continue
            occ2 = occ | {n}
            v = self.score(mi, [first, n], opp, occ2, my_roads, opp_roads, n, None)
            if bv is None or v > bv:
                best, bv = n, v
        return best, bv


def opening_decision(placer, game, actions, me, n_cands=25):
    st = game.state
    mi = map_info(st.board.map)
    opp = st.colors[1 - st.color_to_index[me]]
    mine = list(st.buildings_by_color[me][SETTLEMENT])
    theirs = list(st.buildings_by_color[opp][SETTLEMENT])
    occ = set(mine) | set(theirs)

    def road_map(color, setts):
        rm = {}
        for e in st.buildings_by_color[color].get("ROAD", []):
            a, b = e
            if a in setts:
                rm[a] = b
            elif b in setts:
                rm[b] = a
        return rm
    my_roads = road_map(me, mine)
    opp_roads = road_map(opp, theirs)
    t0 = actions[0].action_type
    if t0 == AT.BUILD_ROAD:
        s = mine[-1]
        best_i, bv = 0, None
        for i, a in enumerate(actions):
            u, w = a.value
            nb = w if u == s else u
            rm = dict(my_roads); rm[s] = nb
            v = placer.score(mi, mine, theirs, occ, rm, opp_roads, mine[1] if len(mine) > 1 else None,
                             theirs[1] if len(theirs) > 1 else None)
            if bv is None or v > bv:
                best_i, bv = i, v
        return best_i
    if t0 != AT.BUILD_SETTLEMENT:
        return None
    idx = {a.value: i for i, a in enumerate(actions)}
    nodes = list(idx)
    if len(mine) == 0:
        def q(n):
            p = mi.node_prod[n]
            return sum(p) + 0.111 * sum(1 for x in p if x > 0)
        cands = sorted(nodes, key=q, reverse=True)[:n_cands]
        best, bv = None, None
        for c in cands:
            occ1 = occ | {c}
            if len(theirs) == 0:      # I am first: opponent takes two, then my second
                o2 = Placer.f_pick(mi, [], occ1, mi.land_nodes)
                occ2 = occ1 | ({o2} if o2 is not None else set())
                o3 = Placer.f_pick(mi, [o2] if o2 is not None else [], occ2, mi.land_nodes)
                occ3 = occ2 | ({o3} if o3 is not None else set())
                onodes = [x for x in (o2, o3) if x is not None]
                m4, v = placer.my_best_second(mi, c, onodes, occ3)
                if m4 is None:
                    continue
                v = placer.score(mi, [c, m4], onodes, occ3 | {m4}, None, None, m4, o3)
            else:                     # I am second: my next pick follows immediately, then theirs
                m3, _ = placer.my_best_second(mi, c, theirs, occ1, opp_roads)
                if m3 is None:
                    continue
                occ2 = occ1 | {m3}
                o4 = Placer.f_pick(mi, theirs, occ2, mi.land_nodes)
                onodes = theirs + ([o4] if o4 is not None else [])
                occ3 = occ2 | ({o4} if o4 is not None else set())
                v = placer.score(mi, [c, m3], onodes, occ3, None, opp_roads, m3, o4)
            if bv is None or v > bv:
                best, bv = c, v
        return idx[best] if best is not None else None
    # my second settlement
    first = mine[0]
    best, bv = None, None
    for n in nodes:
        occ1 = occ | {n}
        if len(theirs) == 1:          # opponent still to pick its second
            o4 = Placer.f_pick(mi, theirs, occ1, mi.land_nodes)
            onodes = theirs + ([o4] if o4 is not None else [])
            occ2 = occ1 | ({o4} if o4 is not None else set())
            v = placer.score(mi, [first, n], onodes, occ2, my_roads, opp_roads, n, o4)
        else:
            v = placer.score(mi, [first, n], theirs, occ1, my_roads, opp_roads, n, theirs[1] if len(theirs) > 1 else None)
        if bv is None or v > bv:
            best, bv = n, v
    return idx[best] if best is not None else None
