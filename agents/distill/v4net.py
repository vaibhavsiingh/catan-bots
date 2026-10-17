"""Value-function agent: one-ply expectimax over a learned estimate of P(win).

For every legal action we build the successor positions in a world sampled from the observation,
expand chance exactly (dice sums, the development card drawn, the card a robber steals), score
every successor with a small neural network trained on self-play outcomes, and play the action
with the highest expected value. A move that wins the game outright is always taken.
"""
import os
import random

import numpy as np

from catanatron.models.enums import Action, ActionRecord, ActionType

from feats import features, DEVS
import fastcopy

fastcopy.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DECK = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DICE = [((1, 1), 1), ((1, 2), 2), ((1, 3), 3), ((2, 3), 4), ((2, 4), 5), ((3, 4), 6),
        ((4, 4), 5), ((4, 5), 4), ((5, 5), 3), ((5, 6), 2), ((6, 6), 1)]
WIN = 2.0


class Value:
    """Ensemble of MLPs; the estimate is the mean of their logits, squashed."""

    def __init__(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        self.models = []
        for path in paths:
            d = np.load(path)
            n = int(d["nlayers"])
            self.models.append((d["mu"], d["sd"], [(d[f"W{k}"], d[f"b{k}"]) for k in range(n)]))

    def __call__(self, rows):
        x = np.asarray(rows, dtype=np.float64)
        z = 0.0
        for mu, sd, layers in self.models:
            h = (x - mu) / sd
            last = len(layers) - 1
            for k, (W, b) in enumerate(layers):
                h = h @ W + b
                if k < last:
                    np.maximum(h, 0.0, out=h)
            z = z + h[:, 0]
        z = z / len(self.models)
        return 1.0 / (1.0 + np.exp(-z))


class Agent:
    def __init__(self, weights=None, lookahead=None):
        self.rng = random.Random(12345)
        if weights is None:
            weights = os.environ.get("AGENT_WEIGHTS")
            weights = weights.split(",") if weights else sorted(
                os.path.join(HERE, f) for f in os.listdir(HERE)
                if f.startswith("value_w") and f.endswith(".npz"))
        self.value = Value(weights)
        if lookahead is None:
            lookahead = os.environ.get("AGENT_LOOKAHEAD", "1") == "1"
        self.lookahead = lookahead

    # ---- chance expansion -------------------------------------------------------------------
    def _outcomes(self, world, a, me, obs):
        """List of (probability, successor game) for action a."""
        t = a.action_type
        if t == ActionType.ROLL:
            out = []
            for dice, w in DICE:
                g = world.copy()
                g.execute(a, validate_action=False, action_record=ActionRecord(a, dice))
                out.append((w / 36.0, g))
            return out
        if t == ActionType.MOVE_ROBBER and a.value[1] is not None:
            opp = obs.opp_resources
            tot = sum(opp.values())
            out = []
            for r in RES:
                c = opp[r]
                if c > 0:
                    g = world.copy()
                    g.execute(a, validate_action=False, action_record=ActionRecord(a, r))
                    out.append((c / tot, g))
            if out:
                return out
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            unseen = {}
            for d in DEVS:
                unseen[d] = (DEV_DECK[d] - obs.my_devs[d] - obs.played_devs["me"][d]
                             - obs.played_devs["opp"][d])
            tot = sum(v for v in unseen.values() if v > 0)
            g = world.copy()
            g.execute(a, validate_action=False)
            st = g.state
            i = st.color_to_index[me]
            drawn = st.action_records[-1].result
            out = []
            if tot <= 0:
                return [(1.0, g)]
            for d, c in unseen.items():
                if c <= 0:
                    continue
                if d == drawn:
                    out.append((c / tot, g))
                    continue
                g2 = g.copy()
                ps = g2.state.player_state
                ps[f"P{i}_{drawn}_IN_HAND"] -= 1
                ps[f"P{i}_{d}_IN_HAND"] += 1
                if drawn == "VICTORY_POINT":
                    ps[f"P{i}_ACTUAL_VICTORY_POINTS"] -= 1
                if d == "VICTORY_POINT":
                    ps[f"P{i}_ACTUAL_VICTORY_POINTS"] += 1
                out.append((c / tot, g2))
            return out
        g = world.copy()
        g.execute(a, validate_action=False)
        return [(1.0, g)]

    def _continuations(self, g, me):
        """Successors of my next deterministic action in the same turn (depth-2 lookahead)."""
        out = []
        for a in g.playable_actions:
            t = a.action_type
            if t in (ActionType.END_TURN, ActionType.BUY_DEVELOPMENT_CARD, ActionType.ROLL):
                continue
            if t == ActionType.MOVE_ROBBER or t.name.startswith("PLAY_"):
                continue
            h = g.copy()
            h.execute(a, validate_action=False)
            out.append(h)
        return out

    def scores(self, obs, actions, world=None, lookahead=None):
        me = obs.me
        if lookahead is None:
            lookahead = self.lookahead and (obs.cpu_left is None or obs.cpu_left > 0.08)
        if world is None:
            world = obs.sample_world(self.rng)
        rows, owner, probs, sub = [], [], [], []
        fixed = {}
        for k, a in enumerate(actions):
            try:
                outs = self._outcomes(world, a, me, obs)
            except Exception:
                fixed[k] = -1.0
                continue
            for p, g in outs:
                grp = len(rows)
                if g.winning_color() == me:
                    rows.append(None)
                else:
                    rows.append(features(g, me))
                    if (lookahead and len(outs) == 1 and a.action_type != ActionType.END_TURN
                            and g.state.current_color() == me
                            and g.state.current_prompt.name == "PLAY_TURN"):
                        for h in self._continuations(g, me):
                            sub.append(grp)
                            rows.append(None if h.winning_color() == me else features(h, me))
                owner.append((k, grp))
                probs.append(p)
        real = [r for r in rows if r is not None]
        v = self.value(real) if real else []
        it = iter(v)
        flat = [WIN if r is None else next(it) for r in rows]
        best = {}
        n_main = len(owner)
        main_idx = [grp for _, grp in owner]
        for grp in main_idx:
            best[grp] = flat[grp]
        # continuation rows follow their parent; parent indices are listed in `sub`
        main_set = set(main_idx)
        cont_rows = [i for i in range(len(rows)) if i not in main_set]
        for i, grp in zip(cont_rows, sub):
            if flat[i] > best[grp]:
                best[grp] = flat[i]
        vals = np.zeros(len(actions))
        for (k, grp), p in zip(owner, probs):
            vals[k] += p * best[grp]
        for k, x in fixed.items():
            vals[k] = x
        return vals

    def decide(self, obs, playable_actions) -> int:
        try:
            vals = self.scores(obs, playable_actions)
            return int(np.argmax(vals))
        except Exception:
            return 0
