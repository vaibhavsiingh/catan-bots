"""Web app for the Catan bots (agents/distill by default), standard library only.

Two modes:
  play     (public)   a 1v1 game against the bot in the browser.
  advisor  (private)  mirror a real game (colonist.io, ...): click the opponent's moves, the bot tells ours.
                      Mounted only with --advisor, and then the server listens on localhost only. A public
                      deployment never serves its pages or its API.

usage:
  python web/server.py [--port 8000]              # public: play against the bot
  python web/server.py --advisor [--port 8001]    # private: advisor (and play), localhost only

The engine is the pinned catanatron; the bot sees what a harness agent sees (Obs over a masked game).
"""

import argparse
import json
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import advise as A  # noqa: E402  board, log and hidden-card logic shared with the terminal advisor
from harness.host import _load_agent  # noqa: E402
from harness.obs import Obs, masked_copy  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.enums import ActionRecord, ActionType  # noqa: E402
from catanatron.models.player import RandomPlayer  # noqa: E402
from catanatron.models.tiles import LandTile, Port  # noqa: E402

AGENT_PATH = os.environ.get("CATAN_AGENT", os.path.join(ROOT, "agents", "distill", "agent.py"))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PRIVATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "private")
GAMES_DIR = os.path.join(ROOT, "games")
MAX_PLAY_GAMES = 200
TURN_LIMIT = 1000
AgentClass = None


def new_agent():
    global AgentClass
    if AgentClass is None:
        AgentClass = _load_agent(AGENT_PATH)
    return AgentClass()


def bot_choice(game, color, agent, rng):
    """the bot's action index, from what a harness agent would see"""
    view = masked_copy(game, color, rng)
    view.state.players = [RandomPlayer(c) for c in view.state.colors]
    view.seed = 0
    i = agent.decide(Obs(view, color), game.playable_actions)
    return i if type(i) is int and 0 <= i < len(game.playable_actions) else 0


# ---------------------------------------------------------------- what the browser gets

def geometry(game):
    """static board: tiles, ports, node positions, edges"""
    m = game.state.board.map
    order = A.land_order(m.tiles)
    tiles, nodes, edges = [], {}, set()
    for i, coord in enumerate(order, 1):
        t = m.tiles[coord]
        x, y = A.center(coord)
        tiles.append(dict(i=i, x=round(x, 1), y=round(y, 1), res=t.resource, num=t.number))
        for ref, n in t.nodes.items():
            nodes[n] = [round(x + A.SIZE * A.CORNER[ref][0], 1), round(y + A.SIZE * A.CORNER[ref][1], 1)]
        for e in t.edges.values():
            edges.add(tuple(sorted(e)))
    ports = []
    for j, coord in enumerate(A.port_order(m.tiles), 1):
        p = m.tiles[coord]
        x, y = A.center(coord)
        pn = sorted(set(p.nodes.values()) & m.port_nodes[p.resource] & set(nodes))
        ports.append(dict(i=j, x=round(x, 1), y=round(y, 1), res=p.resource, nodes=pn))
    return dict(size=A.SIZE, tiles=tiles, ports=ports, nodes=nodes, edges=sorted(edges))


def player_view(game, color, full):
    p, k = A.ps(game, color)
    hand = {r: p[f"{k}_{r}_IN_HAND"] for r in A.RESOURCES}
    devs = {d: p[f"{k}_{d}_IN_HAND"] for d in A.DEVS.values()}
    out = dict(vp=p[f"{k}_VICTORY_POINTS"], cards=sum(hand.values()), devs=sum(devs.values()),
               knights=p[f"{k}_PLAYED_KNIGHT"], road=p[f"{k}_HAS_ROAD"], army=p[f"{k}_HAS_ARMY"],
               longest=p[f"{k}_LONGEST_ROAD_LENGTH"],
               left=dict(roads=p[f"{k}_ROADS_AVAILABLE"], settlements=p[f"{k}_SETTLEMENTS_AVAILABLE"],
                         cities=p[f"{k}_CITIES_AVAILABLE"]))
    if full:
        out["hand"] = hand
        out["devlist"] = {d: n for d, n in devs.items() if n}
        out["actual_vp"] = p[f"{k}_ACTUAL_VICTORY_POINTS"]
    return out


def action_view(game, i, a, steal_known):
    """one playable action: what to draw and what the click needs"""
    t, v = a.action_type, a.value
    d = dict(i=i, type=t.name, desc=A.describe(game, a))
    if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
        d["node"] = v
    elif t == ActionType.BUILD_ROAD:
        d["edge"] = sorted(v)
    elif t == ActionType.MOVE_ROBBER:
        d["tile"] = A.tile_index(game, v[0])
        d["victim"] = None if v[1] is None else A.NAME[v[1]]
        if steal_known and v[1] is not None:
            p, k = A.ps(game, v[1])
            d["steal"] = [r for r in A.RESOURCES if p[f"{k}_{r}_IN_HAND"] > 0]
    elif t == ActionType.MARITIME_TRADE:
        give = [r for r in v[:4] if r is not None]
        d["give"], d["n"], d["get"] = give[0], len(give), v[4]
    elif t == ActionType.DISCARD_RESOURCE:
        d["res"] = v
    elif t == ActionType.PLAY_MONOPOLY:
        d["res"] = v
    elif t == ActionType.PLAY_YEAR_OF_PLENTY:
        d["res2"] = [r for r in v if r]
    return d


def record_text(game, rec):
    a, r = rec.action, rec.result
    who = A.NAME.get(a.color, "?")
    t = a.action_type
    if t == ActionType.ROLL:
        return f"{who} rolled {r[0] + r[1]}"
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return f"{who} bought a development card"
    if t == ActionType.MOVE_ROBBER:
        coord, victim = a.value
        s = f"{who} moved the robber to #{A.tile_index(game, coord)}"
        return s + (f" and stole from {A.NAME[victim]}" if victim else "")
    if t == ActionType.END_TURN:
        return f"{who} ended the turn"
    return f"{who}: {A.describe(game, a)}"


def state_view(game, me, mode, extra=None):
    st = game.state
    cur = st.current_color()
    other = A.THEM if me == A.US else A.US
    rolls = [r.result for r in st.action_records if r.action.action_type == ActionType.ROLL]
    winner = game.winning_color()
    out = dict(
        mode=mode, step=len(st.action_records), turn=st.num_turns,
        current=A.NAME[cur], prompt=st.current_prompt.name,
        robber=A.tile_index(game, st.board.robber_coordinate),
        buildings=[dict(node=n, color=A.NAME[c], kind=kind) for n, (c, kind) in st.board.buildings.items()],
        roads=[dict(edge=[a, b], color=A.NAME[c]) for (a, b), c in st.board.roads.items() if a < b],
        players={A.NAME[me]: player_view(game, me, True),
                 A.NAME[other]: player_view(game, other, mode == "advisor")},
        bank=dict(zip(A.RESOURCES, st.resource_freqdeck)), devdeck=len(st.development_listdeck),
        last_roll=list(rolls[-1]) if rolls else None,
        log=[record_text(game, r) for r in st.action_records[-40:]],
        winner=A.NAME[winner] if winner else None,
    )
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- play mode (public)

class PlayGame:
    """the human plays red (US), the bot blue (THEM)"""

    HUMAN, BOT = A.US, A.THEM

    def __init__(self, first):
        self.lock = threading.Lock()
        self.rng = random.Random(secrets.randbits(64))
        self.mask_rng = random.Random(self.rng.random())
        want = {"you": self.HUMAN, "bot": self.BOT}.get(first) or self.rng.choice([self.HUMAN, self.BOT])
        while True:
            g = Game([RandomPlayer(A.US), RandomPlayer(A.THEM)], seed=self.rng.randrange(2 ** 31))
            if g.state.colors[0] == want:
                break
        self.game = g
        self.agent = new_agent()
        self.touched = time.time()

    def outcome(self, a):
        """dice and steals from our own generator: neither player's search can move them"""
        t = a.action_type
        if t == ActionType.ROLL:
            return (self.rng.randint(1, 6), self.rng.randint(1, 6))
        if t == ActionType.MOVE_ROBBER and a.value[1] is not None:
            p, k = A.ps(self.game, a.value[1])
            cards = [r for r in A.RESOURCES for _ in range(p[f"{k}_{r}_IN_HAND"])]
            return self.rng.choice(cards) if cards else None
        return None

    def apply(self, a):
        r = self.outcome(a)
        if r is None and a.action_type != ActionType.MOVE_ROBBER:
            self.game.execute(a)
        else:
            self.game.execute(a, action_record=ActionRecord(a, r))

    def over(self):
        return self.game.winning_color() is not None or self.game.state.num_turns >= TURN_LIMIT

    def view(self):
        g = self.game
        acts = []
        if not self.over() and g.state.current_color() == self.HUMAN:
            acts = [action_view(g, i, a, False) for i, a in enumerate(g.playable_actions)]
        return state_view(g, self.HUMAN, "play", dict(actions=acts, you=A.NAME[self.HUMAN], bot=A.NAME[self.BOT]))


PLAY = OrderedDict()
PLAY_LOCK = threading.Lock()


def play_get(gid):
    with PLAY_LOCK:
        g = PLAY.get(gid)
        if g:
            PLAY.move_to_end(gid)
            g.touched = time.time()
        return g


def play_api(method, parts, body):
    if method == "POST" and parts == ["new"]:
        g = PlayGame(body.get("first", "random"))
        gid = secrets.token_urlsafe(12)
        with PLAY_LOCK:
            PLAY[gid] = g
            while len(PLAY) > MAX_PLAY_GAMES:
                PLAY.popitem(last=False)
        with g.lock:
            return dict(id=gid, geometry=geometry(g.game), state=g.view())
    if len(parts) < 1:
        return None
    g = play_get(parts[0])
    if g is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "no such game (games expire when the server restarts)")
    with g.lock:
        if method == "GET" and len(parts) == 1:
            return dict(geometry=geometry(g.game), state=g.view())
        if method == "POST" and parts[1:] == ["act"]:
            if body.get("step") != len(g.game.state.action_records):
                raise ApiError(HTTPStatus.CONFLICT, "the game moved on; reload")
            if g.over() or g.game.state.current_color() != g.HUMAN:
                raise ApiError(HTTPStatus.CONFLICT, "not your move")
            i = body.get("i")
            if type(i) is not int or not 0 <= i < len(g.game.playable_actions):
                raise ApiError(HTTPStatus.BAD_REQUEST, "bad action")
            g.apply(g.game.playable_actions[i])
            return dict(state=g.view())
        if method == "POST" and parts[1:] == ["bot"]:
            # one bot action per call, so the browser can show them one at a time
            if not g.over() and g.game.state.current_color() == g.BOT:
                acts = g.game.playable_actions
                i = bot_choice(g.game, g.BOT, g.agent, g.mask_rng) if len(acts) > 1 else 0
                g.apply(acts[i])
            return dict(state=g.view())
    return None


# ---------------------------------------------------------------- advisor mode (private)

class AdvisorGame:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(GAMES_DIR, name + ".json")
        self.spec, self.game = A.load(self.path)
        self.agent = new_agent()
        self.mask_rng = random.Random(0)
        self.advice = {}  # number of log entries -> bot's action index
        self.lock = threading.Lock()

    def reload(self):
        self.spec, self.game = A.load(self.path)

    def commit(self, entry):
        try:
            A.apply(self.game, entry)
        except (AssertionError, ValueError, KeyError) as ex:
            self.reload()
            raise ApiError(HTTPStatus.BAD_REQUEST, f"refused: {ex}")
        self.spec["log"].append(entry)
        A.save(self.path, self.spec)

    def view(self):
        g = self.game
        acts = [action_view(g, i, a, True) for i, a in enumerate(g.playable_actions)]
        cur = g.state.current_color()
        for d, a in zip(acts, g.playable_actions):
            t = a.action_type
            if t == ActionType.ROLL:
                d["needs"] = "dice"
            elif t == ActionType.BUY_DEVELOPMENT_CARD and a.color == A.US:
                d["needs"] = "card"
            elif d.get("steal"):
                d["needs"] = "steal"
        advice = None
        if cur == A.US and g.winning_color() is None:
            key = len(self.spec["log"])
            if key not in self.advice:
                acts_ = g.playable_actions
                self.advice[key] = bot_choice(g, A.US, self.agent, self.mask_rng) if len(acts_) > 1 else 0
            advice = self.advice[key]
        p, k = A.ps(g, A.THEM)
        hidden = [d for d in A.DEVS.values() if p[f"{k}_{d}_IN_HAND"] > 0]
        return state_view(g, A.US, "advisor", dict(
            actions=acts, advice=advice, name=self.name, entries=len(self.spec["log"]),
            them_hidden=len(hidden) > 0, first=self.spec["first"]))


ADV = {}
ADV_LOCK = threading.Lock()
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def adv_get(name):
    if not NAME_RE.match(name or ""):
        raise ApiError(HTTPStatus.BAD_REQUEST, "game names: letters, digits, - and _")
    with ADV_LOCK:
        if name not in ADV:
            if not os.path.exists(os.path.join(GAMES_DIR, name + ".json")):
                raise ApiError(HTTPStatus.NOT_FOUND, f"no game {name}")
            ADV[name] = AdvisorGame(name)
        return ADV[name]


def advisor_api(method, parts, body):
    if method == "GET" and parts == ["list"]:
        os.makedirs(GAMES_DIR, exist_ok=True)
        files = sorted((f for f in os.listdir(GAMES_DIR) if f.endswith(".json")),
                       key=lambda f: -os.path.getmtime(os.path.join(GAMES_DIR, f)))
        return dict(games=[f[:-5] for f in files])
    if method == "POST" and parts == ["new"]:
        name = body.get("name") or time.strftime("g%Y%m%d_%H%M")
        if not NAME_RE.match(name):
            raise ApiError(HTTPStatus.BAD_REQUEST, "game names: letters, digits, - and _")
        spec = dict(tiles=body.get("tiles", ""), ports=body.get("ports", ""), first=body.get("first"),
                    vps=int(body.get("vps", 10)), discard=int(body.get("discard", 7)), log=[])
        try:
            assert spec["first"] in ("us", "them"), "who places first?"
            A.parse_tiles(spec["tiles"])
            A.parse_ports(spec["ports"])
            A.new_game(spec)
        except (AssertionError, ValueError, KeyError) as ex:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(ex))
        path = os.path.join(GAMES_DIR, name + ".json")
        if os.path.exists(path):
            raise ApiError(HTTPStatus.CONFLICT, f"{name} exists already")
        os.makedirs(GAMES_DIR, exist_ok=True)
        A.save(path, spec)
        with ADV_LOCK:
            ADV.pop(name, None)
        return dict(name=name)
    if method == "POST" and parts == ["template"]:
        # empty board geometry for the board editor (tiles in reading order, ports clockwise)
        g = Game([RandomPlayer(A.US), RandomPlayer(A.THEM)], seed=0)
        return dict(geometry=geometry(g))
    if not parts:
        return None
    ag = adv_get(parts[0])
    with ag.lock:
        g = ag.game
        if method == "GET" and len(parts) == 1:
            return dict(geometry=geometry(g), state=ag.view())
        if method != "POST":
            return None
        op = parts[1:]
        if op == ["act"]:
            if body.get("entries") != len(ag.spec["log"]):
                raise ApiError(HTTPStatus.CONFLICT, "the game moved on; reload")
            i = body.get("i")
            if type(i) is not int or not 0 <= i < len(g.playable_actions):
                raise ApiError(HTTPStatus.BAD_REQUEST, "bad action")
            a = g.playable_actions[i]
            t = a.action_type
            r = None
            if t == ActionType.ROLL:
                dice = body.get("dice")
                if type(dice) is not int or not 2 <= dice <= 12:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "dice total 2-12")
                r = [max(1, dice - 6), dice - max(1, dice - 6)]
            elif t == ActionType.BUY_DEVELOPMENT_CARD:
                r = A.placeholder(g) if a.color == A.THEM else body.get("card")
                if r not in A.DEVS.values():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "which card?")
            elif t == ActionType.MOVE_ROBBER and a.value[1] is not None:
                p, k = A.ps(g, a.value[1])
                if sum(p[f"{k}_{x}_IN_HAND"] for x in A.RESOURCES):
                    r = body.get("steal")
                    if r not in A.RESOURCES or p[f"{k}_{r}_IN_HAND"] == 0:
                        raise ApiError(HTTPStatus.BAD_REQUEST, "which card was stolen?")
            ag.commit({"a": repr(a), "r": r})
        elif op == ["undo"]:
            if ag.spec["log"]:
                ag.spec["log"].pop()
                A.save(ag.path, ag.spec)
                ag.reload()
        elif op == ["card"]:
            if body.get("card") not in ("KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"):
                raise ApiError(HTTPStatus.BAD_REQUEST, "which card?")
            ag.commit({"card": body["card"]})
        elif op == ["trade"]:
            moves = []
            for who, give in (("us", body.get("us", {})), ("them", body.get("them", {}))):
                other = "them" if who == "us" else "us"
                for res, n in give.items():
                    if res not in A.RESOURCES or type(n) is not int or n < 0:
                        raise ApiError(HTTPStatus.BAD_REQUEST, "bad trade")
                    if n:
                        moves += [(who, res, -n), (other, res, n)]
            if moves:
                ag.commit({"trade": moves})
        elif op == ["adj"]:
            who, res, n = body.get("who"), body.get("res"), body.get("n")
            if who not in ("us", "them") or res not in A.RESOURCES or type(n) is not int:
                raise ApiError(HTTPStatus.BAD_REQUEST, "bad correction")
            ag.commit({"adj": [who, res, n]})
        else:
            return None
        return dict(geometry=geometry(ag.game), state=ag.view())


# ---------------------------------------------------------------- HTTP

class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status, self.msg = status, msg


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


def make_handler(advisor):
    pages = {"/": os.path.join(STATIC, "index.html")}
    if advisor:
        pages["/advisor"] = os.path.join(PRIVATE, "advisor.html")

    class Handler(BaseHTTPRequestHandler):
        server_version = "catanbot"

        def log_message(self, fmt, *args):
            if "/api/" not in (args[0] if args else ""):
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def send(self, status, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def file(self, path):
            ext = os.path.splitext(path)[1]
            if not os.path.isfile(path) or ext not in TYPES:
                return self.send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            with open(path, "rb") as f:
                self.send(HTTPStatus.OK, f.read(), TYPES[ext])

        def route(self, method):
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in pages:
                return self.file(pages[path])
            if method == "GET" and path.startswith("/static/"):
                name = os.path.basename(path)
                return self.file(os.path.join(STATIC, name))
            if method == "GET" and advisor and path.startswith("/private/"):
                return self.file(os.path.join(PRIVATE, os.path.basename(path)))
            parts = [p for p in path.split("/") if p]
            if len(parts) < 2 or parts[0] != "api" or parts[1] not in ("play", "adv") or (parts[1] == "adv" and not advisor):
                return self.send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 20000:
                    return self.send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    assert isinstance(body, dict)
                except (ValueError, AssertionError):
                    return self.send(HTTPStatus.BAD_REQUEST, {"error": "bad json"})
            try:
                api = play_api if parts[1] == "play" else advisor_api
                out = api(method, parts[2:], body)
            except ApiError as ex:
                return self.send(ex.status, {"error": ex.msg})
            except Exception as ex:  # keep the server up; report the failure
                import traceback
                traceback.print_exc()
                return self.send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(ex).__name__}: {ex}"})
            if out is None:
                return self.send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            self.send(HTTPStatus.OK, out)

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None, help="public mode only (default 0.0.0.0); advisor is always localhost")
    ap.add_argument("--advisor", action="store_true", help="also serve the private advisor (localhost only)")
    a = ap.parse_args()
    host = "127.0.0.1" if a.advisor else (a.host or "0.0.0.0")
    port = a.port or (8001 if a.advisor else 8000)
    new_agent()  # load the bot once before serving
    srv = ThreadingHTTPServer((host, port), make_handler(a.advisor))
    print(f"serving on http://{'localhost' if host == '127.0.0.1' else host}:{port}/"
          + ("  (advisor: /advisor)" if a.advisor else ""), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
