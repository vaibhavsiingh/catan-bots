"""Opponent-model search: assume the opponent is a known deterministic bot (default F) and
simulate its exact replies.

For the policy's top-k actions, run n rollouts each: play the action, then continue with our
policy (greedy) for our moves and the opponent model for theirs, until the end of the opponent's
next turn (or the end of the setup phase). Leaves are scored with a ValueNet's P(win).
Common random numbers: rollout j uses the same dice/deck seed for every candidate, so candidates
are compared under identical luck. The dev deck is reshuffled in each rollout (no peeking).

Only searches when the policy is unsure (top prob < conf) to keep the cost down.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from catanatron.models.enums import ActionType
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from search_player import load_vnet, vnet_batch
from tf_player import load_model, policy_inputs

DEV_TYPES = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]  # catanatron bank order


def determinize(state, opp, rng, mode):
    """Replace what we can't see about the opponent with a random guess consistent with public info.
    dev: their unplayed dev cards are redrawn from (their hand + deck); only the count is public.
         (In 2-player, resource hands are fully trackable from public events, so they stay.)
    all: also resample their resource hand, keeping only its size (pessimistic)."""
    ps, k = state.player_state, f"P{state.color_to_index[opp]}_"
    hand = [d for d in DEV_TYPES for _ in range(ps[k + d + "_IN_HAND"])]
    pool = hand + list(state.development_listdeck)
    rng.shuffle(pool)
    new = pool[: len(hand)]
    state.development_listdeck = pool[len(hand):]
    ps[k + "ACTUAL_VICTORY_POINTS"] += new.count("VICTORY_POINT") - hand.count("VICTORY_POINT")
    for d in DEV_TYPES:
        ps[k + d + "_IN_HAND"] = new.count(d)
    if mode == "all":
        n = sum(ps[k + r + "_IN_HAND"] for r in RES)
        avail = [state.resource_freqdeck[i] + ps[k + r + "_IN_HAND"] for i, r in enumerate(RES)]
        cards = rng.sample([i for i, a in enumerate(avail) for _ in range(a)], n)
        for i, r in enumerate(RES):
            ps[k + r + "_IN_HAND"] = cards.count(i)
            state.resource_freqdeck[i] = avail[i] - cards.count(i)


class CounterPlayer(Player):
    def __init__(self, color, ckpt, vnet, k=4, n=4, conf=0.9, max_ticks=80, opp="F", hidden="none",
                 road=0.0, depth2=False, budget_ms=None, d2k=2):
        super().__init__(color)
        self.model, self.ckpt = load_model(ckpt), ckpt
        self.vnet = load_vnet(vnet)
        self.k, self.n, self.conf, self.max_ticks = k, n, conf, max_ticks
        self.opp, self.opp_kind = {}, opp  # color -> opponent model; "F" or "SELF" (our own policy)
        self.searched, self.hidden = 0, hidden
        # ideas from the lookahead agent: Longest Road nudge, 2-step branching in our turn, time budget
        self.road, self.depth2, self.budget_ms, self.d2k = road, depth2, budget_ms, int(d2k)
        self.t_leaf, self.t_roll = 0.005, 0.01  # running estimates (s) of leaf evaluation and rollout cost

    def logits(self, game, acts):
        with torch.no_grad():
            return self.model(*policy_inputs(game, self.color, acts))[0][0]

    def opp_model(self, color):
        if color not in self.opp:
            if self.opp_kind == "SELF":
                from tf_player import TransformerPlayer
                self.opp[color] = TransformerPlayer(color, ckpt=self.ckpt)
            else:
                self.opp[color] = ValueFunctionPlayer(color)
        return self.opp[color]

    def rollout(self, game, actions, seed):
        """actions: the move to try, optionally followed by a follow-up (skipped if not legal here)."""
        if hasattr(game.state, "random"):
            return self._rollout(game, actions, seed)
        # older engine (the pinned build in requirements.txt): dice and steals come from the global
        # `random`, so seed it for common random numbers and put the caller's state back afterwards
        saved = random.getstate()
        random.seed(seed)
        try:
            return self._rollout(game, actions, seed)
        finally:
            random.setstate(saved)

    def _rollout(self, game, actions, seed):
        g = game.copy()
        rng = random.Random(seed)
        g.state.random = rng
        g.random = rng
        deck = list(g.state.development_listdeck)
        rng.shuffle(deck)
        g.state.development_listdeck = deck
        if self.hidden != "none":
            g.state.resource_freqdeck = list(g.state.resource_freqdeck)
            for c in g.state.colors:
                if c != self.color:
                    determinize(g.state, c, rng, self.hidden)
        g.execute(actions[0])
        for b in actions[1:]:
            if g.winning_color() is None and b in g.playable_actions:
                g.execute(b)
        setup = g.state.is_initial_build_phase
        t0 = g.state.num_turns
        for _ in range(self.max_ticks):
            if g.winning_color() is not None:
                break
            if setup and not g.state.is_initial_build_phase:
                break
            if not setup and g.state.num_turns >= t0 + 2:
                break
            acts = g.playable_actions
            c = g.state.current_color()
            if len(acts) == 1:
                a = acts[0]
            elif c == self.color:
                a = acts[int(self.logits(g, list(acts)).argmax())]
            else:
                a = self.opp_model(c).decide(g, acts)
            g.execute(a, validate_action=False)
        return g

    def road_bonus(self, game, acts):
        """1 for a road that lengthens our longest road while the Longest Road race is close."""
        ps = game.state.player_state
        me = f"P{game.state.color_to_index[self.color]}_"
        op = next(f"P{game.state.color_to_index[c]}_" for c in game.state.colors if c != self.color)
        out = np.zeros(len(acts))
        mine, theirs = ps[me + "LONGEST_ROAD_LENGTH"], ps[op + "LONGEST_ROAD_LENGTH"]
        if not (mine >= 3 and mine + 2 >= theirs and ps[me + "ROADS_AVAILABLE"] > 0):
            return out
        for i, a in enumerate(acts):
            if a.action_type == ActionType.BUILD_ROAD:
                g = game.copy()
                g.execute(a)
                out[i] = g.state.player_state[me + "LONGEST_ROAD_LENGTH"] > mine
        return out

    def series(self, game, acts, cand):
        """(candidate index, action list) per simulation series. depth2: a candidate that leaves us a
        choice in our own turn is split into the policy's top-2 follow-ups; its value is the best of them."""
        if not self.depth2:
            return [(ci, [acts[i]]) for ci, i in enumerate(cand)]
        out = []
        for ci, i in enumerate(cand):
            g = game.copy()
            g.state.random = g.random = random.Random(0)
            g.execute(acts[i])
            nxt = list(g.playable_actions)
            if (g.winning_color() is None and g.state.current_color() == self.color
                    and not g.state.is_initial_build_phase and len(nxt) >= 2
                    and all(a.action_type != ActionType.ROLL for a in nxt)):
                top = np.argsort(-self.logits(g, nxt).numpy())[: self.d2k]
                out += [(ci, [acts[i], nxt[j]]) for j in top]
            else:
                out.append((ci, [acts[i]]))
        return out

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        t_end = time.perf_counter() + self.budget_ms / 1000 if self.budget_ms else float("inf")
        acts = list(playable_actions)
        logits = self.logits(game, acts).numpy()
        if self.road:
            logits = logits + self.road * self.road_bonus(game, acts)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        self.last_searched = False
        if p.max() >= self.conf:
            return acts[int(p.argmax())]
        self.searched += 1
        self.last_searched = True
        cand = np.argsort(-p)[: self.k]
        series = self.series(game, acts, cand)
        seeds = [random.randrange(1 << 30) for _ in range(self.n)]
        leaves, owner = [], []
        out_of_time = False
        for m, s in enumerate(seeds):            # round-robin so every series gets a fair share of time
            for j, (ci, seq) in enumerate(series):
                # stop early enough to also evaluate every leaf collected so far (+ a safety margin)
                if time.perf_counter() + 2 * self.t_roll + (len(leaves) + 1) * self.t_leaf > t_end:
                    out_of_time = True
                    break
                t0 = time.perf_counter()
                leaves.append(self.rollout(game, seq, s))
                owner.append(j)
                self.t_roll = 0.9 * self.t_roll + 0.1 * (time.perf_counter() - t0)
            if out_of_time:
                break
        if not leaves:
            return acts[int(p.argmax())]
        t0 = time.perf_counter()
        v = vnet_batch(self.vnet, leaves, self.color, 0.0)
        self.t_leaf = 0.8 * self.t_leaf + 0.2 * (time.perf_counter() - t0) / len(leaves)
        for j, g in enumerate(leaves):
            w = g.winning_color()
            if w is not None:
                v[j] = float(w == self.color)
        tot, cnt = np.zeros(len(series)), np.zeros(len(series))
        np.add.at(tot, owner, v)
        np.add.at(cnt, owner, 1)
        value = np.full(len(cand), -1.0)
        for j, (ci, _) in enumerate(series):
            if cnt[j]:
                value[ci] = max(value[ci], tot[j] / cnt[j])
        if value.min() < 0:                      # some candidate never simulated: trust the policy
            return acts[int(p.argmax())]
        return acts[cand[int(np.argmax(value))]]
