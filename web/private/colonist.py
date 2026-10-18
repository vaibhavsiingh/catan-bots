"""colonist.io game messages -> an advisor game (board, who is first, and the moves seen so far).

usage: python web/private/colonist.py colonist_capture.json [games/NAME.json]

The capture comes from colonist_capture.js (which also streams to the advisor live, see web/server.py). colonist's
messages are msgpack; message id "130" carries the game: type 4 is the full state at the start (or on reconnect),
type 91 a diff after each change. Without an output path it prints the board for the advisor's paste box.

What is read so far: the board, harbours, who places first, the rules (VP, discard limit) and the setup
placements. Later moves are entered in the advisor as before.

colonist geometry (axial hex coordinates x, y; screen y grows downward, like catanatron's z):
  corner (x, y, z): z=0 the north vertex of hex (x, y), z=1 its south vertex
  edge   (x, y, z): z=0 its NW-N edge, z=1 its SW-NW (west) edge, z=2 its S-SW edge
  tile types 0 desert, 1 wood, 2 brick, 3 sheep, 4 wheat, 5 ore; port types 1 = 3:1, 2-6 = 2:1 of those
"""

import base64
import copy
import json
import math
import os
import re
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import advise as A  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402

TILE = {0: None, 1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE"}
PORT = {1: None, 2: "WOOD", 3: "BRICK", 4: "SHEEP", 5: "WHEAT", 6: "ORE"}
S3 = math.sqrt(3)
CORNER_OFF = {0: (0, -1), 1: (0, 1)}
EDGE_OFF = {0: (-S3 / 4, -0.75), 1: (-S3 / 2, 0), 2: (-S3 / 4, 0.75)}


# ---------------------------------------------------------------- msgpack (only what colonist sends)

def unpack(b, i=0):
    c = b[i]
    i += 1
    if c <= 0x7f:
        return c, i
    if c >= 0xe0:
        return c - 256, i
    if c <= 0x8f:
        return _map(b, i, c & 15)
    if c <= 0x9f:
        return _arr(b, i, c & 15)
    if c <= 0xbf:
        return b[i:i + (c & 31)].decode("utf8", "replace"), i + (c & 31)
    if c in (0xc0, 0xc2, 0xc3):
        return {0xc0: None, 0xc2: False, 0xc3: True}[c], i
    if c in (0xc4, 0xc5, 0xc6, 0xd9, 0xda, 0xdb):  # bin / str with a length
        k = {0xc4: 1, 0xc5: 2, 0xc6: 4, 0xd9: 1, 0xda: 2, 0xdb: 4}[c]
        n = int.from_bytes(b[i:i + k], "big")
        raw = b[i + k:i + k + n]
        return (raw.hex() if c < 0xd9 else raw.decode("utf8", "replace")), i + k + n
    if c == 0xca:
        return struct.unpack(">f", b[i:i + 4])[0], i + 4
    if c == 0xcb:
        return struct.unpack(">d", b[i:i + 8])[0], i + 8
    if 0xcc <= c <= 0xd3:
        k = 1 << ((c - 0xcc) % 4)
        return int.from_bytes(b[i:i + k], "big", signed=c >= 0xd0), i + k
    if c in (0xdc, 0xdd):
        k = 2 if c == 0xdc else 4
        return _arr(b, i + k, int.from_bytes(b[i:i + k], "big"))
    if c in (0xde, 0xdf):
        k = 2 if c == 0xde else 4
        return _map(b, i + k, int.from_bytes(b[i:i + k], "big"))
    if 0xd4 <= c <= 0xd8:  # fixext (timestamps)
        n = 1 << (c - 0xd4)
        return None, i + 1 + n
    raise ValueError(f"msgpack byte {c:#x} at {i - 1}")


def _arr(b, i, n):
    out = []
    for _ in range(n):
        v, i = unpack(b, i)
        out.append(v)
    return out, i


def _map(b, i, n):
    out = {}
    for _ in range(n):
        k, i = unpack(b, i)
        out[str(k)], i = unpack(b, i)
    return out, i


def game_messages(capture):
    """the game's messages (id 130), in order"""
    out = []
    for e in capture:
        if e.get("dir") == "in" and e.get("b64"):
            try:
                v, _ = unpack(base64.b64decode(e["b64"]))
            except (ValueError, IndexError):
                continue
            if isinstance(v, dict) and v.get("id") == "130":
                out.append(v["data"])
    return out


# ---------------------------------------------------------------- geometry

def _px(x, y, off):
    return round(S3 * (x + y / 2) + off[0], 3), round(1.5 * y + off[1], 3)


class Board:
    """colonist corner / edge ids -> catanatron node ids / edges, for one catanatron map"""

    def __init__(self, game, map_state):
        node_px = {}
        for coord, tile in game.state.board.map.land_tiles.items():
            cx, cy = A.center(coord)
            for ref, n in tile.nodes.items():
                ox, oy = A.CORNER[ref]
                node_px[(round(cx / A.SIZE + ox, 3), round(cy / A.SIZE + oy, 3))] = n
        self.node = {int(k): node_px[_px(c["x"], c["y"], CORNER_OFF[c["z"]])]
                     for k, c in map_state["tileCornerStates"].items()}
        mid = {}
        for a in node_px.items():
            for b in node_px.items():
                if a[1] < b[1] and abs(math.dist(a[0], b[0]) - 1) < 1e-3:
                    mid[(round((a[0][0] + b[0][0]) / 2, 3), round((a[0][1] + b[0][1]) / 2, 3))] = (a[1], b[1])
        self.edge = {int(k): mid[_px(e["x"], e["y"], EDGE_OFF[e["z"]])] for k, e in map_state["tileEdgeStates"].items()}


def board_spec(start):
    """advisor spec (tiles, ports, first) from a type-4 payload"""
    ms = start["gameState"]["mapState"]
    by_coord = {}
    for t in ms["tileHexStates"].values():
        x, y = t["x"], t["y"]
        by_coord[(x, -x - y, y)] = (TILE[t["type"]], t["diceNumber"] or None)
    tiles = A.initialize_tiles(A.BASE_MAP_TEMPLATE)
    land = A.land_order(tiles)
    assert set(land) == set(by_coord), "not the standard 19-tile board"
    toks = ["de" if by_coord[c][0] is None else f"{A.SHORT[by_coord[c][0]]}{by_coord[c][1]}" for c in land]
    slots = A.port_order(tiles)
    centers = [tuple(v / A.SIZE for v in A.center(c)) for c in slots]
    ports = [None] * 9
    for p in ms["portEdgeStates"].values():
        m = _px(p["x"], p["y"], EDGE_OFF[p["z"]])
        d, i = min((math.dist(m, c), i) for i, c in enumerate(centers))
        assert abs(d - S3 / 2) < 1e-3, f"harbour {p} is not on a standard harbour slot"
        ports[i] = "3" if PORT[p["type"]] is None else A.SHORT[PORT[p["type"]]]
    me, rules = start["playerColor"], start.get("gameSettings") or {}
    return dict(tiles=" ".join(toks), ports=" ".join(ports), first="us" if start["playOrder"][0] == me else "them",
                vps=rules.get("victoryPointsToWin", 10), discard=rules.get("cardDiscardLimit", 7))


# ---------------------------------------------------------------- moves

DEV = {11: "KNIGHT", 12: "VICTORY_POINT", 13: "MONOPOLY", 14: "ROAD_BUILDING", 15: "YEAR_OF_PLENTY"}
RES = {1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE"}
AT = ActionType


class Stop(Exception):
    """colonist did something the engine can't follow from here"""


def deep_merge(a, d):
    """apply one colonist diff: maps merge key by key, None deletes, anything else replaces"""
    for k, v in d.items():
        if v is None:
            a.pop(k, None)
        elif isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)


class Mirror:
    """replays colonist's messages into an advisor game, as advisor log entries.

    Every move goes through the engine as an action (roll with its dice, builds, robber and steal, discards,
    development cards, bank trades, end of turn); player-to-player trades become "trade" entries. Game log
    types (from colonist's messages): 1 bought a development card, 4 setup placement, 5 build, 10 roll, 11 robber
    moved, 14/15/139 steal, 20 development card played, 21 Year of Plenty gain, 55 discard, 74 nothing to steal,
    86 Monopoly result, 115 player trade, 116 bank trade.
    After each message both hands are checked against colonist (ours is visible; in 1v1 theirs is what neither
    we nor the bank hold) and corrected with "adj" entries if they differ, so a slip can't carry forward.
    """

    def __init__(self, start):
        self.start, self.me = start, start["playerColor"]
        self.spec = board_spec(start)
        self.spec["log"] = []
        self.game = A.new_game(self.spec)
        self.cs = copy.deepcopy(start["gameState"])
        self.board = Board(self.game, self.cs["mapState"])
        self.tiles = {int(k): (t["x"], -t["x"] - t["y"], t["y"]) for k, t in self.cs["mapState"]["tileHexStates"].items()}
        self.placed = {}       # ("c" | "e", colonist id) -> 1 settlement or road, 2 city
        self.pending = []      # discards waiting for the engine to ask that player
        self.robber = None     # (color, coordinate): robber moved, steal not seen yet
        self.dev = None        # (color, card): Monopoly / Year of Plenty played, result not seen yet
        self.notes = []

    # -- helpers
    def color(self, c):
        return A.US if c == self.me else A.THEM

    def other(self, color):
        return A.THEM if color == A.US else A.US

    def entry(self, e):
        A.apply(self.game, e)
        self.spec["log"].append(e)

    def act(self, color, kind, match=lambda v: True, result=None, why=""):
        a = next((x for x in self.game.playable_actions if x.color == color and x.action_type == kind and match(x.value)), None)
        if a is None:
            raise Stop(f"{A.NAME[color]} {kind.name.lower()} {why}is not legal here (engine expects "
                       f"{sorted({x.action_type.name for x in self.game.playable_actions})})")
        self.entry({"a": repr(a), "r": result})
        self.flush()

    def can(self, color, kind):
        return any(x.color == color and x.action_type == kind for x in self.game.playable_actions)

    def flush(self):
        """discards the engine now asks for (colonist discards at once, the engine one player at a time)"""
        progress = True
        while progress:
            progress = False
            for i, (color, res) in enumerate(self.pending):
                if self.can(color, AT.DISCARD_RESOURCE):
                    self.pending.pop(i)
                    self.act(color, AT.DISCARD_RESOURCE, lambda v, res=res: v == res or v is None, why=f"{res} ")
                    progress = True
                    break

    def end_robber(self, victim_card=None, nothing=False):
        color, coord = self.robber
        self.robber = None
        acts = [x for x in self.game.playable_actions if x.action_type == AT.MOVE_ROBBER and x.value[0] == coord]
        want = None if nothing or victim_card is None else self.other(color)
        a = next((x for x in acts if x.value[1] == want), None) or (acts[0] if acts and want is None else None)
        if a is None:
            raise Stop(f"robber to {coord} stealing from {want and A.NAME[want]} is not legal here")
        self.entry({"a": repr(a), "r": victim_card if a.value[1] is not None else None})
        self.flush()

    def unplaced(self):
        """pieces on colonist's board (all diffs so far) that the engine doesn't have yet"""
        ms, out = self.cs["mapState"], []
        for k, c in ms["tileCornerStates"].items():
            level = 2 if c.get("buildingType") == 2 else 1 if c.get("buildingType") == 1 else 0
            have = self.placed.get(("c", int(k)), 0)
            if c.get("owner") and level > have:
                if have == 0 and level == 2:  # settlement and city both new
                    out.append(((("c", int(k)), 1), c["owner"], AT.BUILD_SETTLEMENT, self.board.node[int(k)]))
                out.append(((("c", int(k)), level), c["owner"], AT.BUILD_CITY if level == 2 else AT.BUILD_SETTLEMENT,
                            self.board.node[int(k)]))
        for k, e in ms["tileEdgeStates"].items():
            if e.get("owner") and ("e", int(k)) not in self.placed:
                out.append(((("e", int(k)), 1), e["owner"], AT.BUILD_ROAD, self.board.edge[int(k)]))
        return out

    def place(self, owner=None, kind=None, one=False, strict=True):
        """apply unplaced pieces (of this owner and kind), in an order the engine accepts. strict: at least one must go"""
        done = 0
        while True:
            todo = [t for t in self.unplaced() if (owner is None or t[1] == owner) and (kind is None or t[2] == kind)]
            for key, c, k, v in todo:
                m = (lambda x, v=v: tuple(sorted(x)) == tuple(sorted(v))) if k == AT.BUILD_ROAD else (lambda x, v=v: x == v)
                if any(x.color == self.color(c) and x.action_type == k and m(x.value) for x in self.game.playable_actions):
                    self.act(self.color(c), k, m)
                    self.placed[key[0]] = key[1]
                    done += 1
                    break
            else:
                if strict and not done and todo:  # nothing new: placed already from an earlier message
                    what = f"{todo[0][2].name.lower()} at {todo[0][3]}" if todo else f"{kind.name.lower() if kind else 'piece'} (none new on the board)"
                    raise Stop(f"{A.NAME[self.color(owner)] if owner else ''} {what} is not legal here (engine expects "
                               f"{sorted({x.action_type.name for x in self.game.playable_actions})})")
                return
            if one:
                return

    def end_turn_to(self, c):
        if self.game.winning_color() is None and self.game.state.current_color() != self.color(c) \
                and self.can(self.game.state.current_color(), AT.END_TURN):
            self.act(self.game.state.current_color(), AT.END_TURN)

    def resync(self, start):
        """colonist resent the whole state (a reconnect or reload): catch up on pieces and hands from it"""
        self.cs = copy.deepcopy(start["gameState"])
        self.place(strict=False)
        self.reconcile()

    # -- one message
    PIECE = {0: AT.BUILD_ROAD, 2: AT.BUILD_SETTLEMENT, 3: AT.BUILD_CITY}
    MOVES = {1, 4, 5, 10, 20, 55, 115, 116}  # log types that are a player's next move

    def message(self, m):
        if m.get("type") == 4:
            return self.resync(m["payload"])
        if m.get("type") != 91:
            return
        diff = m["payload"].get("diff", {})
        old_devs = list(self._devs())
        deep_merge(self.cs, diff)  # colonist's state after this message; the log entries say how it got there
        new_devs = list(self._devs())
        for x in old_devs:
            if x in new_devs:
                new_devs.remove(x)
        logs = [e["text"] for _, e in sorted(diff.get("gameLogState", {}).items(), key=lambda x: int(x[0]))
                if isinstance(e, dict) and isinstance(e.get("text"), dict)]
        for t in logs:
            typ, c = t.get("type"), t.get("playerColor")
            if typ in self.MOVES:
                if self.robber:
                    self.end_robber(nothing=True)  # the steal never came: nothing to steal
                if self.dev:
                    raise Stop(f"{self.dev[1].lower()} played but colonist didn't say what it took")
            if typ in (4, 5):
                kind = self.PIECE.get(t.get("pieceEnum"))
                if typ == 5:
                    self.end_turn_to(c)
                self.place(owner=c, kind=kind, one=True)
            elif typ == 10:
                self.end_turn_to(c)
                self.act(self.color(c), AT.ROLL, result=[t["firstDice"], t["secondDice"]])
            elif typ == 11:
                coord = self.tiles[self.cs["mechanicRobberState"]["locationTileIndex"]]
                self.robber = (self.color(c), coord)
            elif typ in (14, 15, 139):
                cards = [RES.get(x) for x in t.get("cardEnums", [])]
                if self.robber and cards and cards[0]:
                    self.end_robber(victim_card=cards[0])
            elif typ == 74:
                if self.robber:
                    self.end_robber(nothing=True)
            elif typ == 55:
                self.pending += [(self.color(c), RES[x]) for x in t.get("cardEnums", [])]
                self.flush()
            elif typ == 1:
                self.end_turn_to(c)
                color = self.color(c)
                if color == A.US:
                    if not new_devs:
                        raise Stop("bought a development card but colonist didn't say which")
                    self.act(color, AT.BUY_DEVELOPMENT_CARD, result=DEV[new_devs.pop(0)])
                else:
                    self.act(color, AT.BUY_DEVELOPMENT_CARD, result=A.placeholder(self.game))
            elif typ == 20:
                self.end_turn_to(c)
                color, card = self.color(c), DEV.get(t.get("cardEnum"))
                if card is None or card == "VICTORY_POINT":
                    continue
                if color == A.THEM:
                    self.entry({"card": card})
                if card == "KNIGHT":
                    self.act(color, AT.PLAY_KNIGHT_CARD)
                elif card == "ROAD_BUILDING":
                    self.act(color, AT.PLAY_ROAD_BUILDING)
                else:
                    self.dev = (color, card)
            elif typ == 21 and self.dev and self.dev[1] == "YEAR_OF_PLENTY":
                got = sorted(RES[x] for x in t.get("cardEnums", []))
                color, self.dev = self.dev[0], None
                self.act(color, AT.PLAY_YEAR_OF_PLENTY, lambda v: sorted(v) == got, why=f"{got} ")
            elif typ == 86 and self.dev and self.dev[1] == "MONOPOLY":
                color, self.dev = self.dev[0], None
                self.act(color, AT.PLAY_MONOPOLY, lambda v: v == RES[t["cardEnum"]])
            elif typ == 116:
                self.bank_trade(self.color(c), [RES[x] for x in t["givenCardEnums"]], [RES[x] for x in t["receivedCardEnums"]])
            elif typ == 115:
                who = "us" if c == self.me else "them"
                other = "them" if who == "us" else "us"
                moves = []
                for give, frm, to in ((t["givenCardEnums"], who, other), (t["receivedCardEnums"], other, who)):
                    for x in set(give):
                        n = give.count(x)
                        moves += [(frm, RES[x], -n), (to, RES[x], n)]
                if moves:
                    self.entry({"trade": moves})
        self.place(strict=False)  # Road Building roads, and pieces whose log entry comes later if the engine allows them now
        turn = diff.get("currentState", {}).get("currentTurnPlayerColor")
        if turn is not None and not self.robber and not self.pending and not self.dev:
            self.end_turn_to(turn)
        if not self.robber and not self.pending and not self.dev and not self.unplaced():
            self.reconcile()

    def _devs(self):
        p = self.cs.get("mechanicDevelopmentCardsState", {}).get("players", {}).get(str(self.me), {})
        return p.get("developmentCards", {}).get("cards", [])

    def bank_trade(self, color, given, received):
        for r in received:
            def ok(v, r=r):
                give = [x for x in v[:-1] if x is not None]
                return v[-1] == r and all(give.count(x) <= given.count(x) for x in set(give))
            a = next((x for x in self.game.playable_actions if x.color == color and x.action_type == AT.MARITIME_TRADE and ok(x.value)), None)
            if a is None:
                raise Stop(f"bank trade {given} -> {received} is not legal here")
            for x in a.value[:-1]:
                if x is not None:
                    given.remove(x)
            self.entry({"a": repr(a), "r": None})

    def reconcile(self):
        """hands as colonist has them; differences become 'adj' entries"""
        bank = self.cs.get("bankState", {})
        ours = self.cs["playerStates"][str(self.me)]["resourceCards"]["cards"]
        truth = {"us": {r: ours.count(k) for k, r in RES.items()}}
        if not bank.get("hideBankCards") and "resourceCards" in bank:
            truth["them"] = {r: 19 - bank["resourceCards"].get(str(k), 0) - truth["us"][r] for k, r in RES.items()}
        for who, hand in truth.items():
            p, k = A.ps(self.game, A.US if who == "us" else A.THEM)
            for r, n in hand.items():
                d = n - p[f"{k}_{r}_IN_HAND"]
                if d:
                    self.notes.append(f"step {len(self.spec['log'])}: {who} {r.lower()} {d:+d} to match colonist")
                    self.entry({"adj": [who, r, d]})


def build(msgs):
    """(game name, advisor spec, problem or None) from the messages of one game, or None before it starts"""
    def gid(m):
        return (m["payload"].get("gameSettings") or {}).get("id")
    starts = [k for k, m in enumerate(msgs) if m.get("type") == 4]
    if not starts:
        return None
    i = next(k for k in starts if gid(msgs[k]) == gid(msgs[starts[-1]]))  # this game's first full state
    start = msgs[i]["payload"]
    mirror, problem = Mirror(start), None
    for m in msgs[i + 1:]:
        try:
            mirror.message(m)
        except (Stop, AssertionError, KeyError, ValueError) as ex:
            problem = f"stopped following colonist at step {len(mirror.spec['log'])}: {ex}"
            break
    gid = re.sub(r"[^A-Za-z0-9]", "", str((start.get("gameSettings") or {}).get("id") or "game"))[:20]
    return f"colonist-{time.strftime('%m%d')}-{gid}", mirror.spec, problem, mirror.notes


def merge(path, spec):
    """write spec to path. colonist's moves win: moves entered by hand are kept only where they go beyond
    colonist's and agree with them up to there (e.g. the bot's move clicked before colonist reported it)"""
    if os.path.exists(path):
        old = json.load(open(path))
        same_board = all(old.get(k) == spec[k] for k in ("tiles", "ports", "first"))
        assert same_board, f"{os.path.basename(path)} has another board"
        if old["log"][:len(spec["log"])] == spec["log"]:
            return old
        spec = dict(old, log=spec["log"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    A.save(path, spec)
    return spec


def main():
    msgs = game_messages(json.load(open(sys.argv[1])))
    built = build(msgs)
    assert built, "no game start in this capture: paste the script before the game starts"
    name, spec, problem, notes = built
    print(f"tiles: {spec['tiles']}\nports: {spec['ports']}\nfirst: {spec['first']}\n{len(spec['log'])} log entries")
    for n in notes:
        print("  corrected:", n)
    if problem:
        print("  " + problem)
    if len(sys.argv) > 2:
        merge(sys.argv[2], spec)
        print(f"-> {sys.argv[2]}")


if __name__ == "__main__":
    main()
