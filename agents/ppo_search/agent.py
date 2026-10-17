"""PPO + search agent: PPO v2 policy + opponent-model search (CounterPlayer), with the
search budget taken from the per-game CPU clock instead of a fixed 450 ms.

Search settings as NEW450: top 4 candidates x 4 rollouts, search only when the policy's top move has
probability < 0.9, Longest Road nudge 3, depth-2 follow-ups, opponent modelled as F, leaves scored by
the value net. Each search gets SHARE of the CPU left above RESERVE, at most 450 ms; when that is
under MIN_SEARCH the policy's move is played without search.
"""

import os
import random

import torch

from counter_player import CounterPlayer

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
RESERVE = 0.25      # s of the allowance never spent on search (covers the policy's own cost on a slower machine)
SHARE = 0.25        # share of the rest one search may use
MAX_SEARCH = 0.45   # s, NEW450's budget
MIN_SEARCH = 0.04   # s, below this a search cannot finish a round of rollouts
SEARCH_SETUP = True  # also search the opening placements


class Agent:
    def __init__(self):
        self.p = CounterPlayer(None, os.path.join(HERE, "ppo2_it20.pt"), os.path.join(HERE, "val2.pt"),
                               k=4, n=4, conf=0.9, opp="F", hidden="none", road=3.0, depth2=True,
                               budget_ms=450)
        self.rng = random.Random(0)
        # warm up torch and the search code here: __init__ is not charged to any game
        from catanatron import Game, Color
        from catanatron.players.value import ValueFunctionPlayer
        g = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=1)
        self.p.color = g.state.current_color()
        self.p.decide(g, g.playable_actions)

    def decide(self, obs, playable_actions) -> int:
        game = obs.sample_world(self.rng)
        acts = list(playable_actions)
        self.p.color = obs.me
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        spend = min(MAX_SEARCH, max(0.0, left - RESERVE) * SHARE)
        search = spend >= MIN_SEARCH and (SEARCH_SETUP or not game.state.is_initial_build_phase)
        self.p.conf = 0.9 if search else 0.0          # conf 0: the policy's move, no search
        self.p.budget_ms = 1000 * spend
        a = self.p.decide(game, acts)
        return acts.index(a)
