"""Two experts and one judge. a2 (sub3's backbone) plays every decision. In the main turn, sub4's
engine (vsk_core) proposes its own move too, and the value-net ensemble (v4net) judges: sub4's
move replaces a2's when the net rates it at least ENS_MARGIN higher in P(win), and the net's own
best move replaces it when rated more than MARGIN higher (sub3's rule).
CPU guard: sub4 and the net are consulted only while NET_MIN_CPU of the allowance is left.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import s4_agent  # noqa: E402  (vsk_core installs its copies; v4net below installs fastcopy's, also exact)
import a2        # noqa: E402
import v4net     # noqa: E402

MARGIN = 0.03
ENS_MARGIN = float(os.environ.get("ENS_MARGIN", "0.0"))
NET_MIN_CPU = 0.25


class Agent(a2.Agent):
    def __init__(self):
        super().__init__()
        self.net = v4net.Agent()
        self.s4 = s4_agent.Agent()
        self._obs = None

    def decide(self, obs, actions):
        self._obs = obs
        return super().decide(obs, actions)

    def _main(self, world, me, actions, deadline):
        i = super()._main(world, me, actions, deadline)
        obs = self._obs
        try:
            if obs is None or (obs.cpu_left is not None and obs.cpu_left < NET_MIN_CPU):
                return i
            vals = self.net.scores(obs, actions, world=world)
            j = int(vals.argmax())
            if j != i and vals[j] - vals[i] > MARGIN:
                return j
            k = self.s4._decide(obs, actions, time.process_time())
            if isinstance(k, int) and 0 <= k < len(actions) and k != i and vals[k] - vals[i] > ENS_MARGIN:
                return k
        except Exception:
            pass
        return i
