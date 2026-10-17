"""Play A vs B (seat-swapped) and record per-player strategy stats -> JSON lines.

usage: python strat_stats.py A B GAMES OUT.jsonl
Per player per game: opening pips per resource, final buildings, dev cards, knights, VP breakdown, win.
"""

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from catanatron import Game, Color
from catanatron.models.enums import ActionType, SETTLEMENT, CITY
from catanatron.state_functions import player_key
from arena import make

RES = ["WHEAT", "ORE", "SHEEP", "WOOD", "BRICK"]


def pips(n):
    return 0 if n is None else 6 - abs(7 - n)


def node_pips(m, node):
    out = dict.fromkeys(RES, 0)
    for t in m.adjacent_tiles[node]:
        if t.resource is not None:
            out[t.resource] += pips(t.number)
    return out


def stats(g, color, label):
    st, m = g.state, g.state.board.map
    k = player_key(st, color)
    recs = [r.action for r in st.action_records if r.action.color == color]
    count = lambda t: sum(a.action_type == t for a in recs)
    opening = dict.fromkeys(RES, 0)
    for a in [a for a in recs if a.action_type == ActionType.BUILD_SETTLEMENT][:2]:
        for r, v in node_pips(m, a.value).items():
            opening[r] += v
    b = st.buildings_by_color[color]
    ps = st.player_state
    return dict(bot=label, win=g.winning_color() == color, first=st.colors[0] == color,
                turns=st.num_turns, vp=ps[f"{k}_ACTUAL_VICTORY_POINTS"],
                opening=opening, settlements=len(b[SETTLEMENT]), cities=len(b[CITY]),
                roads=count(ActionType.BUILD_ROAD), devs=count(ActionType.BUY_DEVELOPMENT_CARD),
                knights=ps[f"{k}_PLAYED_KNIGHT"], vp_cards=ps[f"{k}_VICTORY_POINT_IN_HAND"],
                army=ps[f"{k}_HAS_ARMY"], longest=ps[f"{k}_HAS_ROAD"],
                trades=count(ActionType.MARITIME_TRADE))


def run(args):
    a, b, seeds = args
    torch.set_num_threads(1)
    out = []
    for s in seeds:
        swap = s % 2 == 1
        ca, cb = (Color.BLUE, Color.RED) if swap else (Color.RED, Color.BLUE)
        pa, pb = make(a, ca), make(b, cb)
        g = Game([pb, pa] if swap else [pa, pb], seed=s)
        g.play()
        out.append([stats(g, ca, a), stats(g, cb, b)])
    return out


if __name__ == "__main__":
    a, b, n, path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    W = int(os.environ.get("WORKERS", 15))
    with Pool(W) as pool:
        rs = pool.map(run, [(a, b, list(range(i, n, W))) for i in range(W)])
    with open(path, "w") as f:
        for r in rs:
            for game in r:
                f.write(json.dumps(game) + "\n")
    print(f"wrote {sum(map(len, rs))} games to {path}")
