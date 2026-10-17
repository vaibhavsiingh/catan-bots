"""Lookahead agent: a trained policy/value network + a budgeted search.

Interface (harness/): Agent().decide(obs, playable_actions) -> index.

How it decides:
  1. world = obs.sample_world(rng): a complete Game consistent with what we can see (the opponent's
     unplayed development cards are drawn at random; everything else is exact).
  2. Our network (bot/model.py, checkpoints/final.pt) scores the legal moves on that world.
  3. If the network is unsure and the CPU clock allows it, our opponent-aware search (bot/search.py)
     checks the top candidates by simulating the opponent's reply with catanatron's own
     ValueFunction code, under a per-decision budget computed from obs.cpu_left.
  4. The chosen move is mapped back to its index in playable_actions.

CPU clock (harness/match.py): 300 ms + 12.5 ms per decision, at most 1.5 s per game, and random moves
after it runs out. The search budget is what is left after reserving the network's own cost for
every decision still to come, so the clock is never the reason a game is lost.
"""
import gc
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Settings: environment variables, else bb_defaults.json, else the defaults below.
try:
    with open(os.path.join(HERE, "bb_defaults.json")) as _f:
        _DEFAULTS = json.load(_f)
except FileNotFoundError:
    _DEFAULTS = {}


def _env(key, default):
    return os.environ.get(key, _DEFAULTS.get(key, default))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.models.enums import RESOURCES, ActionPrompt, ActionRecord, ActionType  # noqa: E402
from catanatron.state_functions import get_player_freqdeck  # noqa: E402

from bot.config import Config  # noqa: E402
from bot.model import load  # noqa: E402
from bot.search import SEARCH_PROMPTS, SearchPlayer  # noqa: E402
from opp_detect import OppDetector  # noqa: E402

# Tunables (environment variables override them, for experiments with harness.evaluate)
SEARCH_ON = _env("BB_SEARCH", "1") == "1"
PER_DECISION_RESERVE_S = float(_env("BB_RESERVE_MS", "12")) / 1000   # network cost per future decision
EXPECTED_DECISIONS = int(_env("BB_EXPECTED", "70"))                  # decisions per game, upper-ish
MAX_SEARCH_S = float(_env("BB_MAX_SEARCH_MS", "120")) / 1000         # cap per searched decision
OPENING_SEARCH_S = float(_env("BB_OPENING_MS", "60")) / 1000         # per opening placement
SPEND_FRACTION = float(_env("BB_SPEND_FRAC", "0.25"))                 # share of spare CPU per search
MIN_SEARCH_S = 0.015                                                           # below this, don't bother
SAFETY_S = 0.06
# Clock rule: 1 = reserve BB_RESERVE_MS for each of the next BB_EXPECTED decisions;
# 2 = credit the increment every future decision brings, but keep enough for a game of BB_LONG decisions
# at the measured cost of a decision without search.
CLOCK = _env("BB_CLOCK", "1")
LONG_GAME = int(_env("BB_LONG", "170"))                              # games run 20-160 decisions
NET_MULT = float(_env("BB_NET_MULT", "1.5"))                          # margin on that cost (late views cost more)
SAFETY2_S = float(_env("BB_SAFETY2_MS", "80")) / 1000
CPU_BASE_S, CPU_STEP_S, CPU_MAX_S = 0.300, 0.0125, 1.5                          # the harness clock (harness/match.py)
GC_FREEZE = _env("BB_GC_FREEZE", "0") == "1"
# Opponent detection (opp_detect.py): is the opponent F? If not, search with BB_NONF_VARIANT
# applied (e.g. "opp_model=self"), or not at all with BB_NONF_SEARCH=0.
DETECT = _env("BB_DETECT", "0") == "1"
DETECT_CHECKS = int(_env("BB_DETECT_CHECKS", "40"))
NONF_VARIANT = _env("BB_NONF_VARIANT", "")
NONF_SEARCH = _env("BB_NONF_SEARCH", "1") == "1"
SEARCH_VARIANT = _env("BB_SEARCH_VARIANT", "")
# Value-head lookahead: instead of the policy's top move, play each of its top LA_K moves on a copy and
# score the position with our value head (F's one-move lookahead, with a learned judge).
LOOKAHEAD = _env("BB_LOOKAHEAD", "0") == "1"
LA_K = int(_env("BB_LA_K", "12"))
LA_PRIOR = float(_env("BB_LA_PRIOR", "0.033"))     # score = log P(win) + LA_PRIOR * log policy                        # e.g. "partial=1,cpu_clock=1"
# BB_LA_SPECTRUM=1: a move with a random outcome (a roll, a development card bought, a robber steal) is
# judged by its expected value over every outcome, each forced through an ActionRecord, not by one draw.
LA_SPECTRUM = _env("BB_LA_SPECTRUM", "0") == "1"
# BB_LA_SETUP=0: the opening placements use the policy alone, without the lookahead.
LA_SETUP = _env("BB_LA_SETUP", "1") == "1"
# BB_LA_K_SETUP: candidates for the opening placements (0 = every legal one); the 300 ms the clock starts
# with is meant for them. Default: LA_K, as in the rest of the game.
LA_K_SETUP = int(_env("BB_LA_K_SETUP", str(LA_K))) or 10 ** 6
SETUP_PROMPTS = (ActionPrompt.BUILD_INITIAL_SETTLEMENT, ActionPrompt.BUILD_INITIAL_ROAD)
# BB_LA_DEPTH2=1: plan the turn. A candidate that is a bank trade or a development card and leaves us to move
# again is judged by its best line through the policy's top BB_LA_D next moves (what the trade pays for).
LA_DEPTH2 = _env("BB_LA_DEPTH2", "0") == "1"
LA_D = int(_env("BB_LA_D", "2"))
# Clock guard for it: follow-ups only while the CPU left would still finish a game of BB_D2_HORIZON decisions
# at BB_D2_RESERVE_MS per decision (the plain lookahead's cost). The heavy games are the long ones, past
# decision 96, where the allowance stops growing.
D2_HORIZON = int(_env("BB_D2_HORIZON", "130"))
D2_RESERVE_S = float(_env("BB_D2_RESERVE_MS", "8.5")) / 1000
EXTEND = (ActionType.MARITIME_TRADE, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
          ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_KNIGHT_CARD)
# BB_LA_CONF: outside the opening, play the policy's top move without a lookahead when it is at least this
# likely, to save CPU (1 = always look ahead).
LA_CONF = float(_env("BB_LA_CONF", "1"))
DICE =[((s // 2, s - s // 2), (6 - abs(s - 7)) / 36) for s in range(2, 13)]


def _outcomes(world, action):
    """[(game, probability)] after `action` on a copy of `world`. With LA_SPECTRUM, every outcome of a roll,
    a development card draw (from this world's deck) or a steal (from the victim's exact hand); else, or
    if forcing an outcome fails, the one draw the copy makes."""
    t, cases = action.action_type, None
    if LA_SPECTRUM:
        if t == ActionType.ROLL:
            cases = DICE
        elif t == ActionType.BUY_DEVELOPMENT_CARD:
            deck = world.state.development_listdeck
            cases = [(c, deck.count(c) / len(deck)) for c in sorted(set(deck))]
        elif t == ActionType.MOVE_ROBBER and action.value[1] is not None:
            hand = get_player_freqdeck(world.state, action.value[1])
            cases = [(r, k / sum(hand)) for r, k in zip(RESOURCES, hand) if k]
    if cases:
        try:
            out = []
            for result, prob in cases:
                g = world.copy()
                g.execute(action, action_record=ActionRecord(action=action, result=result))
                out.append((g, prob))
            return out
        except Exception:
            pass
    g = world.copy()
    g.execute(action)
    return [(g, 1.0)]


class Agent:
    def __init__(self):
        """Load and warm up everything here: __init__ is not charged to any game."""
        self.rng = random.Random(0)
        variant = _env("BB_VARIANT", "")   # fast_copy assumes our newer engine: off here
        self.cfg = Config.load(variant).with_variant(SEARCH_VARIANT)
        self.cfg_nonf = self.cfg.with_variant(NONF_VARIANT)
        self.detector = OppDetector(DETECT_CHECKS) if DETECT else None
        self.players = {}
        self.decisions = 0
        self.last_turn = None
        self.net_s = 0.005             # CPU of a decision without search, as the clock counts it (measured below)
        self.search_s = 0.0            # CPU spent in searches this game
        self.d2_ok = True              # room on the clock for turn planning (_depth2_ok), set per decision
        warm = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
        p = SearchPlayer(Color.RED, cfg=self.cfg)
        p.scores(warm)                                   # loads the checkpoint, warms torch up
        self.players[Color.RED] = p
        if GC_FREEZE:                                    # the garbage collector stops rescanning the network and
            gc.collect()                                 # the engine's tables: its full passes cost up to ~100 ms
            gc.freeze()

    def _player(self, color):
        if color not in self.players:
            self.players[color] = SearchPlayer(color, cfg=self.cfg)
        return self.players[color]

    def _new_game(self, obs):
        """A new game starts when the turn counter goes back down."""
        if self.last_turn is None or obs.turn < self.last_turn:
            self.decisions = 0
            self.search_s = 0.0
            if self.detector:
                self.detector.new_game()
        self.last_turn = obs.turn

    def _used(self, obs):
        """CPU this game has used so far, from the clock's own rule (n = this decision, 1-based)."""
        return min(CPU_BASE_S + self.decisions * CPU_STEP_S, CPU_MAX_S) - obs.cpu_left

    def _measure(self, obs):
        """The cost of a decision without search, from the clock itself: what this game has used minus
        its searches, per decision so far. It includes the harness's unpickling of the view, which our
        own timers can't see; the clock ticks in 10 ms, so it is only trusted after 20 decisions."""
        if obs.cpu_left is not None and self.decisions > 20:
            self.net_s = max(0.002, (self._used(obs) - self.search_s) / (self.decisions - 1))

    def _search_budget(self, obs, prompt):
        """Seconds we may spend searching this decision, given the CPU still left in this game."""
        if not SEARCH_ON or (self.detector and not self.detector.is_f and not NONF_SEARCH):
            return 0.0
        if obs.cpu_left is None:                         # no clock (e.g. a local test): search normally
            return MAX_SEARCH_S
        if CLOCK == "2":
            to_come = max(25, LONG_GAME - self.decisions)
            net = NET_MULT * max(0.003, self.net_s)
            spare_long = CPU_MAX_S - self._used(obs) - to_come * net - SAFETY2_S
            spare = min(obs.cpu_left - net - SAFETY2_S, spare_long)
        else:
            to_come = max(10, EXPECTED_DECISIONS - self.decisions)
            spare = obs.cpu_left - to_come * PER_DECISION_RESERVE_S - SAFETY_S
        if prompt in (ActionPrompt.BUILD_INITIAL_SETTLEMENT, ActionPrompt.BUILD_INITIAL_ROAD):
            cap = OPENING_SEARCH_S
        else:
            cap = MAX_SEARCH_S
        budget = min(cap, SPEND_FRACTION * spare)
        return budget if budget >= MIN_SEARCH_S else 0.0

    def _lookahead_ok(self, obs):
        """Room on the clock for a lookahead (~5-15 ms), keeping 4 ms for every decision of a long game."""
        if obs.cpu_left is None:
            return True
        to_come = max(25, LONG_GAME - self.decisions)
        return obs.cpu_left > 0.1 and CPU_MAX_S - self._used(obs) - to_come * 0.004 > 0.1

    def _depth2_ok(self, obs):
        """Room on the clock for follow-ups (see D2_HORIZON)."""
        if obs.cpu_left is None:
            return True
        to_come = max(25, D2_HORIZON - self.decisions)
        return obs.cpu_left > 0.1 and CPU_MAX_S - self._used(obs) - to_come * D2_RESERVE_S > 0.1

    def _follow_ups(self, player, g, a):
        """With LA_DEPTH2 and room on the clock, after a trade or development card `a` that leaves us to move:
        the positions after the policy's top LA_D next moves. Else (or if none can be played) []: g is
        judged as it is."""
        if (not LA_DEPTH2 or not self.d2_ok or a.action_type not in EXTEND or g.winning_color() is not None
                or g.state.current_color() != player.color):
            return []
        acts = g.playable_actions
        out = []
        for x in np.argsort(-player.policy_logp(g, acts))[:LA_D]:
            g2 = g.copy()
            try:
                g2.execute(acts[int(x)])
            except Exception:
                continue
            out.append(g2)
        return out

    def _lookahead(self, player, world, actions, k=LA_K, conf=LA_CONF):
        """The policy's top k moves, each played on a copy and judged by the value head from the point of
        view of whoever moves next (as it was trained): log P(we win) + LA_PRIOR * log policy. A move with
        follow-ups (_follow_ups) gets the best of its lines. Skipped when the policy is at least `conf` sure."""
        logp = player.policy_logp(world, actions)
        order = np.argsort(-logp)[:k]
        if len(order) == 1 or (conf < 1 and np.exp(logp[order[0]]) >= conf):
            return actions[int(order[0])]
        model, enc, _ = load(player.cfg.value_model or player.path, world.state.board.map)
        me = player.color
        p = np.zeros(len(order))
        lines = []                                       # per (candidate, outcome): [j, probability, best P(we win)]
        leaves, owner = [], []

        def judge(g, line):                              # a finished game now, any other position in the batch
            w = g.winning_color()
            if w is not None:
                line[2] = max(line[2], float(w == me))
                return
            mover = g.state.current_color()
            leaves.append(enc(g, mover))
            owner.append((line, mover == me))

        for j, i in enumerate(order):
            a = actions[int(i)]
            try:
                outcomes = _outcomes(world, a)
            except Exception:
                continue                                 # p stays 0
            for g, prob in outcomes:
                line = [j, prob, 0.0]
                lines.append(line)
                nexts = self._follow_ups(player, g, a)
                for g2 in nexts:
                    judge(g2, line)
                if not nexts:
                    judge(g, line)
        if leaves:
            for (line, mine), v in zip(owner, SearchPlayer.judge(model, leaves)):
                line[2] = max(line[2], v if mine else 1.0 - v)
        for j, prob, v in lines:
            p[j] += prob * v
        score = np.log(np.clip(p, 1e-4, 1.0)) + LA_PRIOR * logp[order]
        return actions[int(order[int(np.argmax(score))])]

    def decide(self, obs, playable_actions) -> int:
        self._new_game(obs)
        self.decisions += 1
        self._measure(obs)
        world = obs.sample_world(self.rng)
        player = self._player(obs.me)
        if self.detector:
            try:
                self.detector.observe(world, obs.me)
            except Exception:
                self.detector.mismatches = self.detector.max_checks = 0   # broken: stop checking, keep F
            player.cfg = self.cfg if self.detector.is_f else self.cfg_nonf
        actions = world.playable_actions
        prompt = world.state.current_prompt
        try:
            budget = self._search_budget(obs, prompt) if prompt in SEARCH_PROMPTS else 0.0
            if budget > 0:
                player.cfg.budget_ms = 1000 * budget
                t0 = time.process_time()
                action = player.decide(world, actions)
                self.search_s += time.process_time() - t0
            elif LOOKAHEAD and (LA_SETUP or prompt not in SETUP_PROMPTS) and self._lookahead_ok(obs):
                setup = prompt in SETUP_PROMPTS
                self.d2_ok = self._depth2_ok(obs)
                action =self._lookahead(player, world, actions, LA_K_SETUP if setup else LA_K,
                                         1.0 if setup else LA_CONF)
            else:                                        # the network alone (+ the Longest Road nudge)
                action = actions[int(np.argmax(player.policy_logp(world, actions)))]
        except Exception:
            return 0
        for i, a in enumerate(playable_actions):        # the harness's own action objects
            if a == action:
                return i
        for i, a in enumerate(playable_actions):        # same move, compared field by field
            if (a.action_type, a.value) == (action.action_type, action.value):
                return i
        return 0
