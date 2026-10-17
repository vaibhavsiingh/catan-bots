"""Opponent-aware search: our policy proposes moves, short simulations judge them.

For each candidate move a (by default the policy's top K):
    play a -> [depth2: then each of the policy's top-2 follow-ups b] -> finish our turn with our
    policy -> random dice -> the opponent plays its whole turn with the opponent model (by default
    ValueFunction's real code, which is what the reference bot is) -> stop at our next turn ->
    judge that position with our value head.
value(a) = mean p_win over its M simulations (depth2: the best follow-up's mean).
score(a) = logp(a) / beta + log(value(a)); we play the highest score.

The pieces, each replaceable on its own:
    policy_logp()  policy + strategy nudges            candidates()  top K (+ forced tactics)
    pairs()        what to simulate from where         simulate()    one simulation
    OPPONENTS      who plays the opponent inside      judge()       value head on the leaves
    score()        how logp and p_win combine

Confident decisions (top move at least `conf` likely) skip the search. Simulations run round-robin
(simulation 1 of every pair, then simulation 2, ...) under a hard time budget, so when time runs out
every candidate has had a fair share; if one has none we play the policy's top move.
With depth2, simulation m uses dice seed base + m for every pair (common random numbers), so taking
the max over follow-ups compares like with like instead of rewarding lucky dice.
All settings: bot/config.py. Measured gains: RESULTS.md.
"""
import random
import time

import numpy as np
import torch

from catanatron.models.enums import ActionPrompt
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from bot import nudges
from bot.model import load
from bot.policy import ModelPlayer
from bot.tactics import forced_moves

KEYS = ["tile", "node", "edge", "me", "opp", "global"]
SEARCH_PROMPTS = (ActionPrompt.PLAY_TURN, ActionPrompt.MOVE_ROBBER,
                  ActionPrompt.BUILD_INITIAL_SETTLEMENT, ActionPrompt.BUILD_INITIAL_ROAD)
MAX_SIM_STEPS = 300
# The opponent inside simulations (config opp_model). "self" = our own policy.
OPPONENTS = {"value": ValueFunctionPlayer, "alphabeta": AlphaBetaPlayer, "self": None, "vaib": "vaib"}
CHOICES = {"judge": ("own", "vaib", "avg"), "prior": ("own", "vaib"), "rollout": ("own", "vaib"),
           "setup_horizon": ("turn", "full"), "search_prompts": ("core", "all"), "budget_mode": ("fixed", "adaptive"),
           "opp_model": tuple(OPPONENTS), "opp_alt": tuple(OPPONENTS)}


class SearchPlayer(ModelPlayer):
    def __init__(self, color, params=None, path=None, cfg=None):
        super().__init__(color, params, path, cfg)
        for field, allowed in CHOICES.items():
            if getattr(self.cfg, field) not in allowed:
                raise ValueError(f"{field}={getattr(self.cfg, field)!r}; choose from {', '.join(allowed)}")
        self._opp = {}                                          # (model, colour) -> opponent player
        if self.cfg.fast_copy:
            from bot import fastcopy
            fastcopy.install()                                  # process-wide; exact, so harmless to others

    # ---- the policy ------------------------------------------------------------------------
    def prior_logits(self, game, actions):
        """Logits over `actions` from the candidate generator (config prior), before nudges."""
        if self.cfg.prior == "vaib":
            from vaib_port import adapter
            return adapter.policy_logits(game, self.color, actions)
        logits, _ = self.scores(game)
        return logits

    def policy_logp(self, game, actions):
        logits = self.prior_logits(game, actions)
        logits = logits + nudges.total(self.cfg.nudges, game, self.color, actions)
        return logits - np.logaddexp.reduce(logits)

    def _policy_move(self, game, color):
        """Our policy's move for `color` (argmax, no nudges), used inside simulations."""
        actions = game.playable_actions
        if len(actions) == 1:
            return actions[0]
        saved = self.color
        self.color = color
        try:
            logits, _ = self.scores(game)
        finally:
            self.color = saved
        return actions[int(np.argmax(logits))]

    def _our_move(self, game):
        """Our move inside a simulation (config rollout): our policy's argmax, or his."""
        if self.cfg.rollout == "vaib" and len(game.playable_actions) > 1:
            return self._opponent_move(game, self.color, model="vaib")
        return self._policy_move(game, self.color)

    def _opponent_move(self, game, color, alt=False, model=None):
        actions = game.playable_actions
        if len(actions) == 1:
            return actions[0]
        kind = model or (self.cfg.opp_alt if alt else self.cfg.opp_model)
        if OPPONENTS[kind] is None:
            return self._policy_move(game, color)
        if (kind, color) not in self._opp:
            if kind == "vaib":
                from vaib_port import adapter
                self._opp[kind, color] = adapter.policy_player(color)
            else:
                self._opp[kind, color] = OPPONENTS[kind](color)
        return self._opp[kind, color].decide(game, actions)

    # ---- the pieces of a search --------------------------------------------------------------
    def candidates(self, logp, actions, forced):
        """Indices of the moves to search: the policy's top K, then any forced move not among them."""
        cand = np.argsort(-logp)[: self.cfg.k]
        extra = [i for i, a in enumerate(actions) if a in forced and i not in cand]
        return np.concatenate([cand, extra]).astype(cand.dtype) if extra else cand

    def pairs(self, game, actions, cand, follow, base):
        """(candidate index, start position, first action, seeded) for every simulation series.
        Without depth2 a candidate is one series from here. With depth2, a candidate that leaves us
        in our own turn with a choice is paired with the policy's top-2 follow-ups (plus forced ones)."""
        if not self.cfg.depth2:
            return [(ci, game, actions[i], False) for ci, i in enumerate(cand)]
        out = []
        for ci, i in enumerate(cand):
            a = actions[i]
            g1 = game.copy()
            rng = random.Random(base - 1)
            g1.random = rng
            g1.state.random = rng
            g1.execute(a)
            if (g1.winning_color() is None and g1.state.current_color() == self.color
                    and g1.state.current_prompt == ActionPrompt.PLAY_TURN and len(g1.playable_actions) >= 2):
                scores = self.prior_logits(g1, g1.playable_actions)
                nexts = [g1.playable_actions[int(j)] for j in np.argsort(-scores)[:2]]
                nexts += [b for b in g1.playable_actions if b in follow.get(a, ()) and b not in nexts]
                out += [(ci, g1, b, True) for b in nexts]
            else:
                out.append((ci, game, a, True))     # no follow-up: one series from here
        return out

    def simulate(self, game, first_action, seed=None, alt=False):
        """Play first_action, then continue until it is our turn again after the opponent moved
        (setup_horizon=full: and the setup phase is over). Returns (game, win_or_None): win is 1/0 if
        the game ended, else None (needs the judge).
        seed: give the copy its own dice generator (common random numbers); None keeps the shared one.
        alt: the opponent is played by opp_alt instead of opp_model."""
        full_setup = self.cfg.setup_horizon == "full"
        g = game.copy()
        if seed is not None:
            rng = random.Random(seed)
            g.random = rng
            g.state.random = rng
        g.execute(first_action)
        opponent_moved = False
        for _ in range(MAX_SIM_STEPS):
            w = g.winning_color()
            if w is not None:
                return g, float(w == self.color)
            color = g.state.current_color()
            prompt = g.state.current_prompt
            if color == self.color:
                if (opponent_moved and prompt != ActionPrompt.DISCARD
                        and not (full_setup and g.state.is_initial_build_phase)):
                    return g, None                              # our turn again: stop and judge
                g.execute(self._our_move(g))
            else:
                if prompt == ActionPrompt.PLAY_TURN or g.state.is_initial_build_phase:
                    opponent_moved = True                       # its own turn (not just a discard in ours)
                g.execute(self._opponent_move(g, color, alt))
        return g, None

    @staticmethod
    def judge(model, leaves):
        """Value head on encoded leaf positions (our point of view): p_win per leaf."""
        batch = {k: torch.from_numpy(np.stack([x[k] for x in leaves])) for k in KEYS}
        with torch.no_grad():
            _, v = model(batch)
        return torch.sigmoid(v).numpy()

    CRITICAL_PROMPTS = (ActionPrompt.BUILD_INITIAL_SETTLEMENT, ActionPrompt.BUILD_INITIAL_ROAD,
                        ActionPrompt.MOVE_ROBBER)

    def budget_s(self, game):
        """Search time for this decision: budget_ms, or budget_hi_ms for critical ones (budget_mode=adaptive):
        opening placements, robber moves, and any position where either player has 8+ victory points."""
        cfg = self.cfg
        if cfg.budget_mode == "adaptive":
            if game.state.current_prompt in self.CRITICAL_PROMPTS:
                return cfg.budget_hi_ms / 1000
            from catanatron.state_functions import get_actual_victory_points
            if max(get_actual_victory_points(game.state, c) for c in game.state.colors) >= 8:
                return cfg.budget_hi_ms / 1000
        return cfg.budget_ms / 1000

    def score(self, logp_cand, value):
        return logp_cand / self.cfg.beta + np.log(np.clip(value, 1e-4, 1.0))

    # ---- the decision ------------------------------------------------------------------------
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if game.state.current_prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            return super().decide(game, playable_actions)          # declines trades
        cfg = self.cfg
        logp = self.policy_logp(game, playable_actions)
        best = int(np.argmax(logp))
        forced, follow = set(), {}
        if cfg.force and game.state.current_prompt == ActionPrompt.PLAY_TURN:
            forced, follow = forced_moves(game, self.color, cfg.force_cap_ms)
        searchable = cfg.search_prompts == "all" or game.state.current_prompt in SEARCH_PROMPTS
        if not searchable or (np.exp(logp[best]) >= cfg.conf and not forced):
            return playable_actions[best]                          # confident or minor: no search

        model, enc, _ = load(cfg.value_model or self.path, game.state.board.map)   # the judge
        now = time.process_time if cfg.cpu_clock else time.perf_counter
        deadline = now() + self.budget_s(game)
        cand = self.candidates(logp, playable_actions, forced)
        base = random.randrange(2 ** 30) if cfg.depth2 else None
        series = self.pairs(game, playable_actions, cand, follow, base)
        results = np.zeros((len(series), cfg.m))
        done = np.zeros((len(series), cfg.m), bool)
        leaves, owner = [], []
        n_alt = round(cfg.m * cfg.opp_mix)                         # the last n_alt simulations use opp_alt
        for m in range(cfg.m):                                     # round-robin, m-major
            for p, (ci, start, first, seeded) in enumerate(series):
                if now() > deadline:
                    break
                g, win = self.simulate(start, first, seed=base + m if seeded else None, alt=m >= cfg.m - n_alt)
                done[p, m] = True
                if win is not None:
                    results[p, m] = win
                else:
                    leaves.append(g if cfg.judge == "vaib" else (enc(g, self.color), g) if cfg.judge == "avg"
                                  else enc(g, self.color))         # our move there: our view
                    owner.append((p, m))
        self.last_search = (int(done.sum()), done.size)            # for timing reports
        if leaves:
            if cfg.judge == "vaib":
                from vaib_port import adapter
                p_win = adapter.judge(leaves, self.color)
            elif cfg.judge == "avg":                               # mean of both judges
                from vaib_port import adapter
                p_win = 0.5 * (self.judge(model, [e for e, _ in leaves]) + adapter.judge([g for _, g in leaves], self.color))
            else:
                p_win = self.judge(model, leaves)
            for (p, m), pw in zip(owner, p_win):
                results[p, m] = pw
        n = done.sum(1)
        value = np.full(len(cand), -1.0)                           # -1: no finished simulation
        for p, (ci, _, _, _) in enumerate(series):
            if n[p]:
                value[ci] = max(value[ci], results[p].sum() / n[p])
        if value.min() < 0:
            if not cfg.partial or value[int(np.flatnonzero(cand == best)[0])] < 0:
                return playable_actions[best]
            score = np.where(value >= 0, self.score(logp[cand], np.maximum(value, 0.0)), -np.inf)
            return playable_actions[int(cand[int(np.argmax(score))])]   # among the simulated candidates
        return playable_actions[int(cand[int(np.argmax(self.score(logp[cand], value)))])]
