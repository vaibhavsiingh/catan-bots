"""Learned scoring function: F's hand-made features (+ a few extras), weights fitted to predict winning.

gen:   python learned_value.py gen OUT.npz GAMES      positions from PPO-v2-vs-F and F-vs-F games, both
                                                       perspectives; features + outcome + val2 P(win)
fit:   python learned_value.py fit DATA.npz OUT.json   logistic regression on soft val2 targets (and outcome)
play:  arena spec LF:weights.json  -> 1-ply greedy over afterstates, like F, scored by the fitted logit
"""

import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from catanatron import Game, Color
from catanatron.features import build_production_features, reachability_features
from catanatron.models.player import Player
from catanatron.models.enums import SETTLEMENT, CITY
from catanatron.players.value import ValueFunctionPlayer, value_production
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards, player_key,
                                        player_num_dev_cards, player_num_resource_cards)

RES = ["WHEAT", "ORE", "SHEEP", "WOOD", "BRICK"]
NAMES = (["my_vp", "my_actual_vp", "opp_vp", "production", "enemy_production", "reach0", "reach1",
          "buildable_nodes", "my_road_len", "opp_road_len", "road_len_diff", "my_has_road", "opp_has_road",
          "hand_synergy", "hand_cards", "over7", "my_devs", "opp_devs", "my_knights", "opp_knights",
          "knight_diff", "my_has_army", "opp_has_army", "num_tiles", "turn"]
         + [f"my_{r.lower()}" for r in RES] + [f"opp_{r.lower()}" for r in RES])
_prod = build_production_features(True)


def features(game, color):
    st = game.state
    opp = next(c for c in st.colors if c != color)
    k, ko = player_key(st, color), player_key(st, opp)
    ps = st.player_state
    s = _prod(game, color)  # P0 = color, P1 = opponent
    reach = reachability_features(game, color, 2)
    r0 = sum(reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RES)
    r1 = sum(reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RES)
    hand = {r: ps[f"{k}_{r}_IN_HAND"] for r in RES}
    d_city = (max(2 - hand["WHEAT"], 0) + max(3 - hand["ORE"], 0)) / 5
    d_sett = sum(max(1 - hand[r], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4
    n_hand = player_num_resource_cards(st, color)
    b = st.buildings_by_color[color]
    tiles = set()
    for n in b[SETTLEMENT] + b[CITY]:
        tiles.update(st.board.map.adjacent_tiles[n])
    my_lr, opp_lr = get_longest_road_length(st, color), get_longest_road_length(st, opp)
    my_kn, opp_kn = get_played_dev_cards(st, color, "KNIGHT"), get_played_dev_cards(st, opp, "KNIGHT")
    x = [ps[f"{k}_VICTORY_POINTS"], ps[f"{k}_ACTUAL_VICTORY_POINTS"], ps[f"{ko}_VICTORY_POINTS"],
         value_production(s, "P0"), value_production(s, "P1", False), r0, r1,
         len(st.board.buildable_node_ids(color)), my_lr, opp_lr, my_lr - opp_lr,
         ps[f"{k}_HAS_ROAD"], ps[f"{ko}_HAS_ROAD"], (2 - d_city - d_sett) / 2, n_hand, float(n_hand > 7),
         player_num_dev_cards(st, color), player_num_dev_cards(st, opp), my_kn, opp_kn, my_kn - opp_kn,
         ps[f"{k}_HAS_ARMY"], ps[f"{ko}_HAS_ARMY"], len(tiles), min(st.num_turns, 300) / 100]
    x += [s[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RES] + [s[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RES]
    return np.array(x, np.float32)


def gen_games(args):
    n, seed = args
    import torch
    torch.set_num_threads(1)
    from encode import encode_state
    from search_player import load_vnet
    from tf_player import TransformerPlayer
    random.seed(seed)
    vnet = load_vnet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/val2.pt"))
    X, Y, P = [], [], []
    for i in range(n):
        a, b = Color.RED, Color.BLUE
        pa = (TransformerPlayer(a, ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/ppo2_it20.pt"))
              if i % 2 == 0 else ValueFunctionPlayer(a))
        g = Game([pa, ValueFunctionPlayer(b)], seed=seed * 1000 + i)
        recs = []
        while g.winning_color() is None and g.state.num_turns < 1000:
            if not g.state.is_initial_build_phase and random.random() < 0.08:
                for c in (a, b):
                    recs.append((features(g, c), encode_state(g, c), c))
            g.play_tick()
        w = g.winning_color()
        if not recs:
            continue
        enc = [torch.from_numpy(np.stack([r[1][j] for r in recs])) for j in range(5)]
        with torch.no_grad():
            p = torch.sigmoid(vnet(*enc)[0]).numpy()
        for (x, _, c), pw in zip(recs, p):
            X.append(x); Y.append(0.5 if w is None else float(w == c)); P.append(pw)
    return X, Y, P


def fit(X, t, l2=1e-3, iters=4000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - t) / len(t) + l2 * np.r_[w[:-1], 0])
    return w, mu, sd


def logloss(X, y, w, mu, sd):
    p = 1 / (1 + np.exp(-(np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), float(((p > .5) == (y > .5)).mean())


class LearnedValuePlayer(Player):
    def __init__(self, color, path):
        super().__init__(color)
        d = json.load(open(path))
        sd = np.array(d["sd"])
        self.w = np.array(d["w"][:-1]) / sd  # score = w . (x - mu)/sd  (constant dropped)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        best, best_a = -np.inf, None
        for a in playable_actions:
            g = game.copy()
            g.execute(a)
            if g.winning_color() == self.color:
                return a
            v = float(features(g, self.color) @ self.w)
            if v > best:
                best, best_a = v, a
        return best_a


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        out, games = sys.argv[2], int(sys.argv[3])
        W = int(os.environ.get("WORKERS", 14))
        with Pool(W) as pool:
            rs = pool.map(gen_games, [(games // W, random.randrange(10**6) + i) for i in range(W)])
        X = np.array([x for r in rs for x in r[0]]); Y = np.array([y for r in rs for y in r[1]])
        P = np.array([p for r in rs for p in r[2]])
        np.savez_compressed(out, X=X, Y=Y, P=P)
        print(f"{len(X)} positions")
    elif cmd == "fit":
        d = np.load(sys.argv[2])
        X, Y, P = d["X"].astype(np.float64), d["Y"].astype(np.float64), d["P"].astype(np.float64)
        n = len(X); cut = int(n * 0.85)  # sequential split ~ split by game
        res = {}
        for name, t in (("soft", P), ("outcome", Y)):
            w, mu, sd = fit(X[:cut], t[:cut])
            res[name] = (w, mu, sd)
            print(f"fit on {name:7s}: held-out outcome logloss/acc {logloss(X[cut:], Y[cut:], w, mu, sd)}")
        pv = np.clip(P[cut:], 1e-6, 1 - 1e-6)
        print(f"val2 itself      : held-out outcome logloss {-(Y[cut:]*np.log(pv)+(1-Y[cut:])*np.log(1-pv)).mean():.4f}")
        w, mu, sd = res["soft"]
        json.dump({"names": NAMES, "w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}, open(sys.argv[3], "w"))
        for nm, wi in sorted(zip(NAMES, w[:-1]), key=lambda z: -abs(z[1])):
            print(f"  {nm:18s} {wi:+.3f}")
