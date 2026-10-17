"""Expert-iteration data: the search player (policy + rollouts + value net) plays against a mix of
opponents; we record its decisions as policy targets and sample positions for the value net.

usage: python gen_exit.py OUT_PREFIX SECONDS POLICY VALUE
  writes OUT_PREFIX_w{i}.npz (policy: pack() fields + searched) and OUT_PREFIX_v_w{i}.npz (value)
env: WORKERS, OPP_MIX (default "F:0.3,AB:0.2,TF:0.2,SEARCH:0.3"), KEEP_UNSEARCHED (0.2),
     VALUE_P (0.05), SEARCH (k:n:conf, default 4:4:0.9)
"""

import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from catanatron import Game, Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import get_actual_victory_points
from counter_player import CounterPlayer
from encode import encode_state, encode_actions
from gen_data import pack
from tf_player import TransformerPlayer

H = 4  # dvp horizon in turns (same as gen_value.py)
COLORS = [Color.RED, Color.BLUE]


def opp_mix():
    spec = os.environ.get("OPP_MIX", "F:0.3,AB:0.2,TF:0.2,SEARCH:0.3")
    names, w = zip(*[(x.split(":")[0], float(x.split(":")[1])) for x in spec.split(",")])
    return list(names), list(w)


class Recorder(CounterPlayer):
    def __init__(self, color, buf, keep, **kw):
        super().__init__(color, **kw)
        self.buf, self.keep = buf, keep

    def decide(self, game, playable_actions):
        a = super().decide(game, playable_actions)
        if len(playable_actions) > 1 and (self.last_searched or random.random() < self.keep):
            acts = list(playable_actions)
            self.buf.append((self.color, encode_state(game, self.color),
                             encode_actions(game, self.color, acts), acts.index(a), self.last_searched))
        return a


def worker(args):
    wid, seconds, policy, value, prefix = args
    torch.set_num_threads(1)
    seed = random.randrange(1 << 30) + wid
    random.seed(seed)
    k, n, conf = os.environ.get("SEARCH", "4:4:0.9").split(":")
    skw = dict(ckpt=policy, vnet=value, k=int(k), n=int(n), conf=float(conf), hidden="dev")
    keep = float(os.environ.get("KEEP_UNSEARCHED", 0.2))
    vp_p = float(os.environ.get("VALUE_P", 0.05))
    names, w = opp_mix()
    pol, val, games, searched = [], [], 0, 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        opp = random.choices(names, w)[0]
        buf = []
        me, other = random.sample(COLORS, 2)
        players = {me: Recorder(me, buf, keep, opp="F" if opp in ("F", "AB") else "SELF", **skw)}
        if opp == "F":
            players[other] = ValueFunctionPlayer(other)
        elif opp == "AB":
            players[other] = AlphaBetaPlayer(other)
        elif opp == "TF":
            players[other] = TransformerPlayer(other, ckpt=policy)
        else:
            players[other] = Recorder(other, buf, keep, opp="SELF", **skw)
        g = Game([players[c] for c in COLORS], seed=random.randrange(1 << 30))
        vp, vrecs = {}, []
        while g.winning_color() is None and g.state.num_turns < 1000:
            t = g.state.num_turns
            vp[t] = {c: get_actual_victory_points(g.state, c) for c in COLORS}
            if random.random() < vp_p:
                vrecs += [(encode_state(g, c), c, t) for c in COLORS]
            g.play_tick()
        win = g.winning_color()
        end = {c: get_actual_victory_points(g.state, c) for c in COLORS}
        vp[g.state.num_turns] = end
        for c, s, a, y, srch in buf:
            pol.append((s, a, y, 0.5 if win is None else float(win == c), srch))
            searched += srch
        for s, c, t in vrecs:
            o = COLORS[1] if c == COLORS[0] else COLORS[0]
            t2 = min(t + H, g.state.num_turns)
            while t2 not in vp:
                t2 -= 1
            val.append((s, 0.5 if win is None else float(win == c),
                        (vp[t2][c] - vp[t2][o]) - (vp[t][c] - vp[t][o]), (end[c] - end[o]) / 10))
        games += 1
    if pol:
        d = pack([p[:4] for p in pol])
        d["searched"] = np.array([p[4] for p in pol], np.bool_)
        np.savez_compressed(f"{prefix}_w{wid}.npz", **d)
    if val:
        st = lambda i: np.stack([x[0][i] for x in val]).astype(np.float16)
        np.savez_compressed(f"{prefix}_v_w{wid}.npz", tile=st(0), node=st(1), edge=st(2), player=st(3),
                            glob=st(4), win=np.array([x[1] for x in val], np.float16),
                            dvp=np.array([x[2] for x in val], np.float16),
                            margin=np.array([x[3] for x in val], np.float16))
    return games, len(pol), searched, len(val)


if __name__ == "__main__":
    prefix, seconds, policy, value = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    W = int(os.environ.get("WORKERS", 15))
    with Pool(W) as pool:
        rs = pool.map(worker, [(i, seconds, policy, value, prefix) for i in range(W)])
    g, p, s, v = map(sum, zip(*rs))
    print(f"{g} games, {p} policy samples ({s} searched), {v} value samples", flush=True)
