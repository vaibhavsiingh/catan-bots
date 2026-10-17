"""Play a real 1v1 game (colonist.io, a physical board, ...) with our bot's advice.

You sit in for the bot: the script mirrors the game in catanatron, asks the bot (agents/distill by default) for
our moves, and you enter what happens: the board, the dice, the opponent's moves and the cards drawn or stolen.
The bot sees what a harness agent sees (the opponent's unplayed development cards are hidden from it).

usage:
  python tools/advise.py --template              # writes advise_template.html: tile and port slots
  python tools/advise.py GAME.json [--agent agents/distill/agent.py]

GAME.json is created on first run (you are asked for the board) and every step is saved to it, so you can quit
and resume. A live picture of the board (node numbers, roads, robber) is written next to it, GAME.html; keep it
open in a browser, it refreshes itself.

Board input, in reading order of the picture in advise_template.html (top row left to right, rows 3-4-5-4-3):
  tiles: 19 tokens, resource code + number: wo wood, br brick, sh sheep, wh wheat, or ore, de desert (no number)
         e.g.  wh6 or8 sh5  br10 wo3 wh9 sh12  ...  de ...
  ports: 9 tokens for slots P1..P9 (clockwise from the top): 3 for a 3:1 port, or a resource code for 2:1.
Check that your board's harbours sit where the template's slots are.

At each prompt:
  our move    enter = done as advised; a number = you did something else (see `list`)
  their move  a number from the list, or words to filter it (e.g. `road 12`, `settle`, `end`)
  any time    list | hands | u (undo the last step) | q (quit, progress is saved)
              card knight|road|yop|mono      the opponent is about to play this development card
              trade us:wo2,br1 them:wh1      a trade between players (each side gives what follows it)
              adj us|them RES N              correct a hand (N may be negative; the bank makes up the difference)
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # noqa: E402,F401  the determinism fix
from harness.host import _load_agent  # noqa: E402
from harness.obs import Obs, masked_copy  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.enums import ActionRecord, ActionType  # noqa: E402
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, initialize_tiles  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402
from catanatron.models.tiles import LandTile, Port  # noqa: E402
from catanatron.models.enums import NodeRef  # noqa: E402

US, THEM = Color.RED, Color.BLUE
NAME = {US: "US", THEM: "THEM"}
CODES = {"wo": "WOOD", "br": "BRICK", "sh": "SHEEP", "wh": "WHEAT", "or": "ORE"}
SHORT = {v: k for k, v in CODES.items()}
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEVS = {"knight": "KNIGHT", "road": "ROAD_BUILDING", "yop": "YEAR_OF_PLENTY", "mono": "MONOPOLY", "vp": "VICTORY_POINT"}
PLACEHOLDER_ORDER = ["KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY", "VICTORY_POINT"]
TILE_COUNTS = {"WOOD": 4, "BRICK": 3, "SHEEP": 4, "WHEAT": 4, "ORE": 3, None: 1}
NUMBERS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
SIZE = 40
CORNER = {NodeRef.NORTH: (0, -1), NodeRef.NORTHEAST: (0.866, -0.5), NodeRef.SOUTHEAST: (0.866, 0.5),
          NodeRef.SOUTH: (0, 1), NodeRef.SOUTHWEST: (-0.866, 0.5), NodeRef.NORTHWEST: (-0.866, -0.5)}


# ---------------------------------------------------------------- board

def center(coord):
    x, _, z = coord
    return math.sqrt(3) * SIZE * (x + z / 2), 1.5 * SIZE * z


def land_order(tiles):
    """land coordinates in reading order (rows top to bottom, left to right)"""
    land = [c for c, t in tiles.items() if isinstance(t, LandTile)]
    return sorted(land, key=lambda c: (round(center(c)[1]), center(c)[0]))


def port_order(tiles):
    """port coordinates clockwise from 12 o'clock"""
    ports = [c for c, t in tiles.items() if isinstance(t, Port)]
    return sorted(ports, key=lambda c: math.atan2(center(c)[0], -center(c)[1]) % (2 * math.pi))


def parse_tiles(text):
    toks = text.split()
    assert len(toks) == 19, f"need 19 tiles, got {len(toks)}"
    out = []
    for t in toks:
        letters, digits = re.sub(r"\d", "", t).lower(), re.sub(r"\D", "", t)  # wo12 or 12wo
        if letters in ("de", "desert", "ro", "robber"):
            out.append((None, None))
            continue
        assert letters[:2] in CODES and digits, f"can't read {t!r}: use e.g. wo12 (or 12wo), and de for the desert"
        assert 2 <= int(digits) <= 12 and int(digits) != 7, f"no {digits} token in Catan ({t!r})"
        out.append((CODES[letters[:2]], int(digits)))
    res = [r for r, _ in out]
    for r, k in TILE_COUNTS.items():
        if res.count(r) != k:
            print(f"  warning: {res.count(r)} {r or 'desert'} tiles (standard board: {k})")
    if sorted(n for _, n in out if n is not None) != NUMBERS:
        print("  warning: the numbers are not the standard set")
    return out


def parse_ports(text):
    toks = text.split()
    assert len(toks) == 9, f"need 9 ports, got {len(toks)}"
    for t in toks:
        assert t == "3" or t[:2].lower() in CODES, f"can't read port {t!r}: use 3 or wo/br/sh/wh/or"
    return [None if t == "3" else CODES[t[:2].lower()] for t in toks]


def build_map(spec):
    tiles = initialize_tiles(BASE_MAP_TEMPLATE)
    for coord, (res, num) in zip(land_order(tiles), parse_tiles(spec["tiles"])):
        tiles[coord].resource, tiles[coord].number = res, num
    for coord, res in zip(port_order(tiles), parse_ports(spec["ports"])):
        tiles[coord].resource = res
    return CatanMap.from_tiles(tiles)


def new_game(spec):
    first = US if spec["first"] == "us" else THEM
    for seed in range(1000):  # the engine shuffles the seating; take a seed that seats `first` first
        game = Game([RandomPlayer(US), RandomPlayer(THEM)], seed=seed, catan_map=build_map(spec),
                    vps_to_win=spec.get("vps", 10), discard_limit=spec.get("discard", 7))
        if game.state.colors[0] == first:
            return game
    raise RuntimeError("could not seat the first player")


def tile_label(tile):
    return "desert" if tile.resource is None else f"{SHORT[tile.resource]}{tile.number}"


def node_label(game, n):
    m = game.state.board.map
    tiles = "/".join(tile_label(t) for t in m.adjacent_tiles[n])
    port = next((("3:1" if r is None else f"{SHORT[r]} 2:1") for r, ns in m.port_nodes.items() if n in ns), None)
    return f"{n} ({tiles}{', port ' + port if port else ''})"


def tile_index(game, coord):
    return land_order(game.state.board.map.tiles).index(coord) + 1


def describe(game, a):
    t, v = a.action_type, a.value
    if t == ActionType.BUILD_SETTLEMENT:
        return f"settlement at node {node_label(game, v)}"
    if t == ActionType.BUILD_CITY:
        return f"city at node {node_label(game, v)}"
    if t == ActionType.BUILD_ROAD:
        return f"road {v[0]}-{v[1]}"
    if t == ActionType.MOVE_ROBBER:
        coord, victim = v
        tile = game.state.board.map.land_tiles[coord]
        return f"robber to tile #{tile_index(game, coord)} ({tile_label(tile)})" + (f", steal from {NAME[victim]}" if victim else "")
    if t == ActionType.MARITIME_TRADE:
        give = [r for r in v[:4] if r is not None]
        return f"bank trade {len(give)} {SHORT[give[0]]} -> 1 {SHORT[v[4]]}"
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return "play year of plenty: " + " + ".join(SHORT[r] for r in v if r)
    if t == ActionType.PLAY_MONOPOLY:
        return f"play monopoly: {SHORT[v]}"
    if t == ActionType.DISCARD_RESOURCE:
        return f"discard {SHORT[v]}"
    return {ActionType.ROLL: "roll", ActionType.END_TURN: "end turn", ActionType.BUY_DEVELOPMENT_CARD: "buy development card",
            ActionType.PLAY_KNIGHT_CARD: "play knight", ActionType.PLAY_ROAD_BUILDING: "play road building"}.get(t, repr(a))


# ---------------------------------------------------------------- hidden cards

def ps(game, color):
    st = game.state
    return st.player_state, f"P{st.color_to_index[color]}"


def placeholder(game):
    """the card we give the opponent when they buy one (we can't see it; the bot never sees it either)"""
    deck = game.state.development_listdeck
    return next(c for c in PLACEHOLDER_ORDER if c in deck)


def swap_hidden(game, old, new):
    """the opponent's hidden card `old` is really `new`: exchange it with the deck"""
    p, k = ps(game, THEM)
    deck = game.state.development_listdeck
    deck.remove(new)
    deck.append(old)
    p[f"{k}_{old}_IN_HAND"] -= 1
    p[f"{k}_{new}_IN_HAND"] += 1
    p[f"{k}_ACTUAL_VICTORY_POINTS"] += (new == "VICTORY_POINT") - (old == "VICTORY_POINT")


def free_card(game, card):
    """make sure `card` is in the deck (one of the opponent's placeholders may hold it)"""
    if card in game.state.development_listdeck:
        return
    p, k = ps(game, THEM)
    assert p[f"{k}_{card}_IN_HAND"] > 0, f"no {card} left anywhere"
    swap_hidden(game, card, placeholder(game))


def reveal(game, card):
    """the opponent is about to play `card`"""
    p, k = ps(game, THEM)
    if p[f"{k}_{card}_IN_HAND"] == 0:
        old = next((c for c in PLACEHOLDER_ORDER if c != card and p[f"{k}_{c}_IN_HAND"] > 0), None)
        assert old is not None, "the opponent holds no unplayed development card"
        assert card in game.state.development_listdeck, f"no {card} left in the deck"
        swap_hidden(game, old, card)
    p[f"{k}_{card}_OWNED_AT_START"] = True


# ---------------------------------------------------------------- the log

def apply(game, e):
    """one log entry: {"a": repr(action), "r": result} or {"card"|"trade"|"adj": ...}"""
    if "card" in e:
        reveal(game, e["card"])
        game.playable_actions = generate_playable_actions(game.state)
        return
    if "trade" in e or "adj" in e:
        st = game.state
        moves = e["trade"] if "trade" in e else [e["adj"]]
        for who, res, n in moves:
            p, k = ps(game, US if who == "us" else THEM)
            p[f"{k}_{res}_IN_HAND"] += n
            if "adj" in e:
                i = RESOURCES.index(res)
                st.resource_freqdeck[i] -= n
        game.playable_actions = generate_playable_actions(game.state)
        return
    a = next((x for x in game.playable_actions if repr(x) == e["a"]), None)
    assert a is not None, f"not playable now: {e['a']}"
    r = e.get("r")
    if a.action_type == ActionType.ROLL:
        r = tuple(r)
    if a.action_type == ActionType.BUY_DEVELOPMENT_CARD and a.color == US:
        free_card(game, r)
    game.execute(a, action_record=ActionRecord(a, r))


def load(path):
    spec = json.load(open(path))
    game = new_game(spec)
    for e in spec["log"]:
        apply(game, e)
    return spec, game


def save(path, spec):
    with open(path + ".tmp", "w") as f:
        json.dump(spec, f, indent=1)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- picture

FILL = {"WOOD": "#5b8c3a", "BRICK": "#b5562f", "SHEEP": "#a8d26a", "WHEAT": "#e8c547", "ORE": "#8a8f99", None: "#d9c79c"}


def render(game, path, template=False):
    m = game.state.board.map
    pos = {}
    parts = []
    for i, coord in enumerate(land_order(m.tiles), 1):
        t = m.tiles[coord]
        cx, cy = center(coord)
        pts = " ".join(f"{cx + SIZE * dx:.1f},{cy + SIZE * dy:.1f}" for dx, dy in CORNER.values())
        parts.append(f'<polygon points="{pts}" fill="{"#eee" if template else FILL[t.resource]}" stroke="#555"/>')
        label = f"#{i}" if template else f"#{i} {tile_label(t)}"
        parts.append(f'<text x="{cx}" y="{cy + 4}" class="t">{label}</text>')
        if not template and game.state.board.robber_coordinate == coord:
            parts.append(f'<circle cx="{cx}" cy="{cy + 16}" r="7" fill="#222"/>')
        for ref, n in t.nodes.items():
            pos[n] = (cx + SIZE * CORNER[ref][0], cy + SIZE * CORNER[ref][1])
    for j, coord in enumerate(port_order(m.tiles), 1):
        p = m.tiles[coord]
        cx, cy = center(coord)
        label = f"P{j}" if template else ("3:1" if p.resource is None else f"{SHORT[p.resource]} 2:1")
        parts.append(f'<text x="{cx}" y="{cy + 4}" class="p">{label}</text>')
        for n in p.nodes.values():
            if n in pos:
                parts.append(f'<line x1="{cx}" y1="{cy}" x2="{pos[n][0]}" y2="{pos[n][1]}" stroke="#4a7fb5" stroke-dasharray="3,3"/>')
    if not template:
        col = {US: "#d33", THEM: "#36c"}
        for (a, b), c in game.state.board.roads.items():
            if a < b:
                parts.append(f'<line x1="{pos[a][0]}" y1="{pos[a][1]}" x2="{pos[b][0]}" y2="{pos[b][1]}" stroke="{col[c]}" stroke-width="6"/>')
        for n, (c, kind) in game.state.board.buildings.items():
            r = 11 if kind == "CITY" else 8
            parts.append(f'<rect x="{pos[n][0] - r}" y="{pos[n][1] - r}" width="{2 * r}" height="{2 * r}" fill="{col[c]}"/>')
    for n, (x, y) in pos.items():
        parts.append(f'<text x="{x}" y="{y + 3}" class="n">{n}</text>')
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    vb = f"{min(xs) - 70} {min(ys) - 70} {max(xs) - min(xs) + 140} {max(ys) - min(ys) + 140}"
    title = "Board template" if template else "Game board"
    note = ("Enter tiles #1-#19 in this order and ports P1-P9 clockwise from the top."
            if template else "Red = us, blue = them. Squares: settlements (small) and cities (large). Black dot: robber.")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
{'' if template else '<meta http-equiv="refresh" content="2">'}
<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#222}} svg{{width:100%;max-width:820px}}
.t{{font-size:11px;text-anchor:middle;font-weight:bold}} .p{{font-size:11px;text-anchor:middle;fill:#24578f;font-weight:bold}}
.n{{font-size:8px;text-anchor:middle;fill:#000;paint-order:stroke;stroke:#fff;stroke-width:2px}}</style></head>
<body><p>{note}</p><svg viewBox="{vb}">{''.join(parts)}</svg></body></html>"""
    with open(path, "w") as f:
        f.write(html)


# ---------------------------------------------------------------- prompts

def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        sys.exit(0)


def until_valid(prompt, check):
    """ask until `check` accepts the answer"""
    while True:
        text = ask(prompt)
        try:
            check(text)
            return text
        except (AssertionError, ValueError, KeyError, ZeroDivisionError) as ex:
            print(f"  {ex if isinstance(ex, AssertionError) else 'try again'}")


def hands(game):
    for c in (US, THEM):
        p, k = ps(game, c)
        hand = " ".join(f"{SHORT[r]}{p[f'{k}_{r}_IN_HAND']}" for r in RESOURCES)
        devs = sum(p[f"{k}_{d}_IN_HAND"] for d in DEVS.values())
        mine = ""
        if c == US:
            mine = " (" + ", ".join(f"{n} {p[f'{k}_{d}_IN_HAND']}" for n, d in DEVS.items() if p[f"{k}_{d}_IN_HAND"]) + ")"
        print(f"  {NAME[c]:4s} VP {p[f'{k}_VICTORY_POINTS']:2d}  hand {hand}  dev cards {devs}{mine if devs else ''}"
              f"  knights played {p[f'{k}_PLAYED_KNIGHT']}")


def outcome(game, a):
    """the random result of an action, asked from the user; None for deterministic actions"""
    t = a.action_type
    if t == ActionType.ROLL:
        while True:
            s = ask("  dice (total, or both dice): ").split()
            try:
                v = [int(x) for x in s]
                if len(v) == 1 and 2 <= v[0] <= 12:
                    return [max(1, v[0] - 6), v[0] - max(1, v[0] - 6)]
                if len(v) == 2 and all(1 <= x <= 6 for x in v):
                    return v
            except ValueError:
                pass
            print("  e.g. 8, or 3 5")
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        if a.color == THEM:
            return placeholder(game)
        while True:
            s = ask("  which card did you get (knight / vp / road / yop / mono)? ").lower()
            if s in DEVS:
                return DEVS[s]
    if t == ActionType.MOVE_ROBBER and a.value[1] is not None:
        victim = a.value[1]
        p, k = ps(game, victim)
        if sum(p[f"{k}_{r}_IN_HAND"] for r in RESOURCES) == 0:
            return None
        q = "  what did you steal" if a.color == US else "  what did they steal from you"
        while True:
            s = ask(f"{q} (wo/br/sh/wh/or)? ").lower()[:2]
            if s in CODES and p[f"{k}_{CODES[s]}_IN_HAND"] > 0:
                return CODES[s]
            print(f"  {NAME[victim]} has no {s}" if s in CODES else "  wo, br, sh, wh or or")
    return None


def parse_trade(words):
    """trade us:wo2,br1 them:wh1 -> [(who, RES, n)] with the givers' losses and the takers' gains"""
    moves = []
    for w in words:
        who, _, items = w.partition(":")
        other = "them" if who == "us" else "us"
        for it in filter(None, items.split(",")):
            res, n = CODES[it[:2]], int(it[2:] or 1)
            moves += [(who, res, -n), (other, res, n)]
    return moves


def command(path, spec, game, line):
    """shared commands; returns the new game, or None if `line` is not a command"""
    w = line.split()
    if not w:
        return None
    if w[0] == "q":
        save(path, spec)
        sys.exit(0)
    if w[0] == "hands":
        hands(game)
        return game
    if w[0] == "u":
        if spec["log"]:
            print(f"  undone: {spec['log'].pop()}")
            save(path, spec)
        return load(path)[1]
    entry = None
    if w[0] == "card" and len(w) == 2 and w[1] in DEVS:
        entry = {"card": DEVS[w[1]]}
    elif w[0] == "trade" and len(w) >= 2:
        entry = {"trade": parse_trade(w[1:])}
    elif w[0] == "adj" and len(w) == 4 and w[1] in ("us", "them") and w[2][:2] in CODES:
        entry = {"adj": [w[1], CODES[w[2][:2]], int(w[3])]}
    if entry is None:
        return None
    try:
        apply(game, entry)
    except (AssertionError, ValueError, KeyError) as ex:
        print(f"  refused: {ex}")
        return load(path)[1]
    spec["log"].append(entry)
    save(path, spec)
    return game


def matches(query, text):
    """every query word starts some word of the text: `road 20` finds road 20-22, not road 1-120"""
    split = lambda t: [w for w in re.split(r"[\s\-/(),:#]+", t.lower()) if w]  # noqa: E731
    words = split(text)
    return all(any(w.startswith(q) for w in words) for q in split(query))


def show(game, acts, mark=None):
    for i, a in enumerate(acts):
        print(f"  {'*' if i == mark else ' '}{i:3d}  {describe(game, a)}")


def step(path, spec, game, agent, rng):
    acts = game.playable_actions
    color = game.state.current_color()
    if color == US:
        view = masked_copy(game, US, rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        view.seed = 0
        t0 = time.process_time()
        i = agent.decide(Obs(view, US), acts)
        print(f"\n>>> BOT ({1000 * (time.process_time() - t0):.0f} ms): {describe(game, acts[i]).upper()}")
        while True:
            line = ask("  enter = done, number = you did something else, list: ")
            if line.lower() in ("", "done", "d", "y", "yes", "ok"):
                break
            if line == "list":
                show(game, acts, i)
                continue
            if line.isdigit() and int(line) < len(acts):
                i = int(line)
                break
            g = command(path, spec, game, line)
            if g is not None:
                return g
        a = acts[i]
    else:
        print(f"\n--- THEM: what did they do? ({len(acts)} options; `list` shows them)")
        if len(acts) <= 12:
            show(game, acts)
        while True:
            line = ask("  > ")
            if line == "list":
                show(game, acts)
                continue
            if line.isdigit() and int(line) < len(acts):
                a = acts[int(line)]
                break
            g = command(path, spec, game, line)
            if g is not None:
                return g
            hits = [k for k, x in enumerate(acts) if matches(line, describe(game, x))]
            if len(hits) == 1:
                a = acts[hits[0]]
                print(f"  = {describe(game, a)}")
                break
            for k in hits[:40]:
                print(f"   {k:3d}  {describe(game, acts[k])}")
            if not hits:
                print("  nothing matches; `list`, `card ...` if they play a development card, `u` to undo")
    r = outcome(game, a)
    entry = {"a": repr(a), "r": r}
    apply(game, entry)
    spec["log"].append(entry)
    save(path, spec)
    return game


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?")
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "distill", "agent.py"))
    ap.add_argument("--template", action="store_true")
    a = ap.parse_args()
    if a.template:
        tiles = initialize_tiles(BASE_MAP_TEMPLATE)
        g = Game([RandomPlayer(US), RandomPlayer(THEM)], seed=0, catan_map=CatanMap.from_tiles(tiles))
        render(g, "advise_template.html", template=True)
        print(f"wrote {os.path.abspath('advise_template.html')}")
        return
    if not a.game:
        ap.error("give GAME.json (or --template)")
    path = os.path.abspath(a.game)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    html = os.path.splitext(path)[0] + ".html"
    if not os.path.exists(path):
        print("New game. Tiles in reading order (see advise_template.html), e.g. wh6 or8 sh5 ...")
        spec = {"tiles": until_valid("tiles: ", parse_tiles),
                "ports": until_valid("ports P1..P9 (3 or wo/br/sh/wh/or): ", parse_ports),
                "first": until_valid("who places first, us or them? ", lambda t: t.lower() in ("us", "them") or 1 / 0).lower(),
                "vps": int(ask("victory points to win [10]: ") or 10),
                "discard": int(ask("discard when holding more than [7]: ") or 7), "log": []}
        new_game(spec)
        save(path, spec)
    spec, game = load(path)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(a.agent)))  # agents load their files relative to themselves
    agent = _load_agent(a.agent)()
    os.chdir(cwd)
    rng = random.Random(0)
    print(f"board picture: {html} (open it in a browser; it refreshes itself)")
    while True:
        render(game, html)
        last = spec["log"][-1].get("a", "") if spec["log"] else "AT.END_TURN"
        if any(t in last for t in ("AT.END_TURN", "AT.ROLL", "AT.MOVE_ROBBER", "AT.DISCARD")):
            hands(game)
        w = game.winning_color()
        if w is not None:
            print(f"\n{NAME[w]} reached {spec.get('vps', 10)} VP (by the engine's count). `u` to undo, `q` to quit.")
            g = command(path, spec, game, ask("> "))
            game = g or game
            continue
        game = step(path, spec, game, agent, rng)


if __name__ == "__main__":
    main()
